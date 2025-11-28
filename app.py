import os
import json

from typing import List, Dict, Any
import re


from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np

# -------------------------
# Flask setup
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"  # <- change in production
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)


# -------------------------
# Models
# -------------------------

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Upload(db.Model):
    __tablename__ = "uploads"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default="completed")

    header_json = db.Column(db.Text)      # JSON string
    components_json = db.Column(db.Text)  # JSON string
    nodes_json = db.Column(db.Text)       # JSON string
    details_json = db.Column(db.Text)
    user = db.relationship("User", backref=db.backref("uploads", lazy=True))


# -------------------------
# Auth helpers
# -------------------------

def login_required(view_func):
    """
    Authentication temporarily disabled:
    this decorator now just calls the view directly.
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        return view_func(*args, **kwargs)
    return wrapper


def current_user():
    """
    Single-user mode: always return a default demo user.
    If none exists, create it.
    """
    user = User.query.first()
    if user is None:
        user = User(
            email="demo@example.com",
            password_hash=generate_password_hash("demo"),
        )
        db.session.add(user)
        db.session.commit()
    return user



# -------------------------
# Excel parsing helpers
# -------------------------

def get_header(df):
    header = {}
    for i in range(df.shape[0]):
        row = list(df.iloc[i])
        if "Vendor Number" in row:
            j = row.index("Vendor Number")
            header["vendor_number"] = row[j + 1]
        if "Item" in row and "item" not in header:
            j = row.index("Item")
            header["item"] = row[j + 1]
        if "Item Number" in row:
            j = row.index("Item Number")
            header["item_number"] = row[j + 1]
        if "Ship Date:" in row:
            j = row.index("Ship Date:")
            header["ship_date"] = row[j + 1]
        if "PO Quantity:" in row:
            j = row.index("PO Quantity:")
            header["po_quantity"] = row[j + 1]
    # convert dates to string for JSON
    if isinstance(header.get("ship_date"), (pd.Timestamp, datetime)):
        header["ship_date"] = header["ship_date"].strftime("%Y-%m-%d")
    return header




def normalize_percent(val):
    """Convert excel percent values to a 0–100 float, or None."""
    if pd.isna(val):
        return None

    # String like "100%" or "20"
    if isinstance(val, str):
        s = val.strip().replace("%", "")
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f

    # Numeric
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None

    # Excel-style 1.0 => 100%, 0.82 => 82%
    if 0 <= f <= 1:
        return f * 100.0
    return f


def get_components(df):
    components = []

    # --------- FORMAT A: Vertical "Component Breakdown" (old template) ---------
    for i in range(df.shape[0]):
        row = list(df.iloc[i])
        if "Component Breakdown" in row:
            j = row.index("Component Breakdown")
            comp_col = j
            perc_col = j + 1
            origin_col = j + 2
            remarks_col = j + 3

            for k in range(i + 1, df.shape[0]):
                name = df.iat[k, comp_col]

                # stop when component cell is blank
                if (
                    name is None
                    or (isinstance(name, float) and pd.isna(name))
                    or (isinstance(name, str) and not name.strip())
                ):
                    break

                percent = df.iat[k, perc_col] if perc_col < df.shape[1] else None
                origin = df.iat[k, origin_col] if origin_col < df.shape[1] else None
                remarks = df.iat[k, remarks_col] if remarks_col < df.shape[1] else None

                components.append(
                    {
                        "name": str(name).strip(),
                        "percent": normalize_percent(percent),
                        "origin": str(origin).strip() if pd.notna(origin) else None,
                        "remarks": str(remarks).strip() if pd.notna(remarks) else None,
                    }
                )

            if components:
                return components  # Found and parsed this format

    # --------- FORMAT B: Horizontal "Item Major Fabric Breakdown" (Pinewood) ---------
    for i in range(df.shape[0]):
        row = list(df.iloc[i])

        header_col = None
        for idx, val in enumerate(row):
            if isinstance(val, str) and "Major Fabric Breakdown" in val:
                header_col = idx   # this is the COMPONENT column
                break

        if header_col is None:
            continue

        comp_col = header_col

        # In the same row we have "%", "Origin Countries", "Remarks:"
        perc_col = origin_col = remarks_col = None
        for c in range(header_col + 1, df.shape[1]):
            label = df.iat[i, c]
            if not isinstance(label, str):
                continue
            text = label.strip().lower()
            if text == "%" or text.endswith("%"):
                perc_col = c
            elif text.startswith("origin"):
                origin_col = c
            elif text.startswith("remarks"):
                remarks_col = c

        # Now read data from the rows below
        for k in range(i + 1, df.shape[0]):
            name = df.iat[k, comp_col]
            if not isinstance(name, str) or not name.strip():
                break  # stop when component column becomes blank

            percent = df.iat[k, perc_col] if perc_col is not None else None
            origin = df.iat[k, origin_col] if origin_col is not None else None
            remarks = df.iat[k, remarks_col] if remarks_col is not None else None

            components.append(
                {
                    "name": name.strip(),
                    "percent": normalize_percent(percent),
                    "origin": str(origin).strip() if pd.notna(origin) else None,
                    "remarks": str(remarks).strip() if pd.notna(remarks) else None,
                }
            )

        break  # handled this header row

    return components



def nearest_value(df, row_idx, col_idx, window=1):
    """
    Try to pick a non-empty cell from [col_idx-window, col_idx+window]
    at the given row index.
    """
    for off in range(0, window + 1):
        for sign in (0,) if off == 0 else (+1, -1):
            c = col_idx + sign * off
            if 0 <= c < df.shape[1]:
                val = df.iat[row_idx, c]
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if not isinstance(val, str) and pd.notna(val):
                    return str(val)
    return None


def get_nodes(df):
    """
    Parse 'Document Group 1..7' horizontal flow from the sample template.
    This is tailored to the current training template layout.
    """
    nodes = []

    # --- helpers ---
    def _clean(val):
        return val.strip() if isinstance(val, str) and val.strip() else None

    def _combine(name, loc):
        """
        Combine company name + location:
        ABC Company + Daqing Heilongjiang -> 'ABC Company, Daqing Heilongjiang'
        """
        name = _clean(name)
        loc = _clean(loc)
        if name and loc:
            if name in loc:  # avoid duplication if name already inside location
                return loc
            return f"{name}, {loc}"
        return name or loc or None


    def _looks_like_quantity(text: str) -> bool:
        """Heuristic to detect lines that are actually quantities."""
        if not isinstance(text, str):
            return False
        t = text.lower()

        # contains unit / quantity words
        if any(u in t for u in [
            "kg", "kilogram", "metric ton", "metric tons", "mt",
            "bale", "bales", "tons", "ton"
        ]):
            return True

        # fallback: at least one digit
        return any(ch.isdigit() for ch in t)




    # rows where different fields live (based on your file)
    row_material = 9
    row_company_type = 11
    row_company_name = 12
    row_location = 13
    row_remarks = 15
    row_date = 16
    row_qty = 17
    docs_start = 18
    docs_end = 27

    # find "Document Group X" cells
    coords_dg = []
    for j in range(df.shape[1]):
        v = df.iat[row_company_type, j]
        if isinstance(v, str) and v.startswith("Document Group"):
            coords_dg.append((row_company_type, j, v))

    # to chain left ← previous right
    previous_right_type = None
    previous_right_party = None

    for _, col, dg_label in coords_dg:
        material = df.iat[row_material, col]

        # -------- LEFT BOX --------
        # For DG2, DG3, ... left side is the right side of previous group
        if previous_right_party:
            left_type = previous_right_type
            left_party = previous_right_party
        else:
            # Only DG1 reads left box directly from Excel
            left_type = df.iat[row_company_type, col - 1] if col - 1 >= 0 else None
            left_name = df.iat[row_company_name, col - 1] if col - 1 >= 0 else None
            left_loc  = df.iat[row_location,    col - 1] if col - 1 >= 0 else None
            left_party = _combine(left_name, left_loc)

        # -------- RIGHT BOX --------
        right_type = df.iat[row_company_type, col + 1] if col + 1 < df.shape[1] else None
        right_name = df.iat[row_company_name, col + 1] if col + 1 < df.shape[1] else None
        right_loc  = df.iat[row_location,    col + 1] if col + 1 < df.shape[1] else None
        right_party = _combine(right_name, right_loc)

        # other fields (same as before)
        # Remarks can still come from left/right
        remarks = nearest_value(df, row_remarks, col, window=1)



        # ----- QUANTITY: only use the value in the DG column itself -----
        raw_qty = df.iat[row_qty, col] if row_qty < df.shape[0] else None
        if (
            isinstance(raw_qty, (int, float, np.integer, np.floating))
            and not pd.isna(raw_qty)
        ):
            # pretty formatting (no ".0" if integer)
            if isinstance(raw_qty, float) and raw_qty.is_integer():
                quantity = str(int(raw_qty))
            else:
                quantity = str(raw_qty)
        elif isinstance(raw_qty, str) and raw_qty.strip() and any(ch.isdigit() for ch in raw_qty):
            quantity = raw_qty.strip()
        else:
            # cell empty or only text like "Manufacturer Production Records"
            quantity = "none"


        # ----- DATE: search for a real date in this DG column -----


        def _looks_like_date(val):
            if isinstance(val, (pd.Timestamp, datetime)):
                return True
            if not isinstance(val, str):
                return False
            s = val.strip()
            if not s:
                return False
            if not any(ch.isdigit() for ch in s):
                return False
            if not any(ch in "/-" for ch in s):
                return False
            return True

        date = "none"

        # Merged-cell fix: search row_date..row_date+2 across nearby columns
        for r in range(row_date, min(row_date + 3, df.shape[0])):
            for c in (col, col - 1, col + 1):
                if 0 <= c < df.shape[1]:
                    val = df.iat[r, c]

                    if isinstance(val, (pd.Timestamp, datetime)):
                        date = val.strftime("%Y-%m-%d")
                        break

                    if _looks_like_date(val):
                        date = str(val).strip()
                        break
            if date != "none":
                break

        # If row contains a long sentence, treat as remarks, not date
        if isinstance(date, str) and len(date) > 40:
            if not remarks:
                remarks = date
            date = "none"



        docs = []
        for r in range(docs_start, docs_end + 1):
            v = df.iat[r, col]
            if isinstance(v, str) and v.strip():
                # Split multi-line cells (e.g. "Contract\nInvoice")
                for part in v.splitlines():
                    part = part.strip()
                    if part:
                        docs.append(part)


        # ------------------------------------------------------------------
        # Wool template fixes: clean "date" and recover quantity
        # ------------------------------------------------------------------

        # 1) If the "date" cell is actually a long sentence like
        #    "Goods are shipped from the Sheep Farm ...", treat it as remarks,
        #    not as a real date.
        if isinstance(date, str):
            s = date.strip()
            if s:
                lower = s.lower()
                if "goods are shipped" in lower or len(s) > 40:
                    # move to remarks if we don't have one yet
                    if not remarks:
                        remarks = s
                    date = "none"

        # 2) If quantity is empty / "none" but the first document line looks
        #    like a quantity (e.g. "92 Metric Tons Wool"), promote it.
        if (not quantity or str(quantity).strip().lower() == "none") and docs:
            first_doc = docs[0]
            if _looks_like_quantity(first_doc):
                quantity = first_doc
                docs = docs[1:]
        # ------------------------------------------------------------------


        nodes.append(
            {
                "group": dg_label,
                "material": _clean(material),

                # left box shown as header + content
                "left_type": _clean(left_type),
                "left_party": left_party,

                # right box shown as header + content
                "right_type": _clean(right_type),
                "right_party": right_party,

                "date": date,
                "quantity": quantity,
                "remarks": remarks,
                "documents": docs,
            }
        )

        # chain this right to next group's left
        previous_right_type = right_type
        previous_right_party = right_party

    return nodes


def get_detail_blocks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Parse the bottom 'Production ... Records' blocks.

    This version is tolerant to extra blank / arrow rows, so it works for
    Plastics, Pinewood and Wool templates.
    """
    blocks: List[Dict[str, Any]] = []

    # Any "(Something)" is considered a candidate block header
    title_pattern = re.compile(r"^\(.+?\)$")

    max_row, max_col = df.shape

    # small helper to safely read a cell
    def safe(r: int, c: int):
        if 0 <= r < max_row and 0 <= c < max_col:
            return df.iat[r, c]
        return None

    def looks_like_date(val) -> bool:
        if isinstance(val, (pd.Timestamp, datetime)):
            return True
        if not isinstance(val, str):
            return False
        s = val.strip()
        if not s:
            return False
        if not any(ch.isdigit() for ch in s):
            return False
        if not any(ch in "/-" for ch in s):
            return False
        return True

    for col in range(max_col):
        for row in range(max_row):
            cell = df.iat[row, col]
            if not isinstance(cell, str):
                continue

            raw_title = cell.strip()
            # Only proceed if the cell looks like "(Role Name)"
            if not title_pattern.match(raw_title):
                continue

            block_type = raw_title.strip("()").strip()

            name = safe(row + 1, col)
            location = safe(row + 2, col)

            # ---------- DATE: search in a small vertical window ----------
            date = "none"
            for r in range(row + 4, min(row + 9, max_row)):
                v = safe(r, col)
                if isinstance(v, (pd.Timestamp, datetime)):
                    date = v.strftime("%Y-%m-%d")
                    break
                if looks_like_date(v):
                    date = str(v).strip()
                    break

            # ---------- DOC TITLE + DOCUMENTS (flexible scan) ----------
            doc_title: str | None = None
            documents: List[str] = []

            # Start a bit above where the old code started and skip blanks
            r = row + 5
            while r < max_row:
                v = safe(r, col)
                if isinstance(v, str) and v.strip():
                    break
                r += 1

            # Now from first non-empty line downward until a blank
            while r < max_row:
                v = safe(r, col)
                if not isinstance(v, str) or not v.strip():
                    break
                text = v.strip()
                if doc_title is None and "record" in text.lower():
                    # first "…records" line → title
                    doc_title = text
                else:
                    documents.append(text)
                r += 1

            # Fallback: if still no title but first document has "records"
            if doc_title is None and documents:
                first = documents[0]
                if isinstance(first, str) and "record" in first.lower():
                    doc_title = first
                    documents = documents[1:]

            # ---- FILTER: skip only truly empty blocks ----
            if not doc_title and not documents:
                continue

            blocks.append(
                {
                    "type": block_type,
                    "name": name,
                    "location": location,
                    "date": date,
                    "doc_title": doc_title,
                    "documents": documents,
                }
            )

    return blocks





def parse_supply_chain_excel(file_path: str):
    df = pd.read_excel(file_path, sheet_name=0, header=None)
    header = get_header(df)
    components = get_components(df)
    nodes = get_nodes(df)
    details = get_detail_blocks(df)
    return header, components, nodes, details



# -------------------------
# Routes: auth
# -------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please log in.", "error")
            return redirect(url_for("login"))

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        session["user_id"] = user.id
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------------
# Routes: main app
# -------------------------

@app.route("/")
@login_required
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
# @login_required
def dashboard():
    user = current_user()
    uploads = (
        Upload.query.filter_by(user_id=user.id)
        .order_by(Upload.created_at.desc())
        .all()
    )

    total = len(uploads)
    completed = sum(1 for u in uploads if u.status == "completed")
    failed = sum(1 for u in uploads if u.status == "failed")

    return render_template(
        "dashboard.html",
        user=user,
        uploads=uploads,
        total=total,
        completed=completed,
        failed=failed,
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please select an Excel file.", "error")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(save_path)

        try:
            header, components, nodes, detail_blocks = parse_supply_chain_excel(save_path)


            upload_row = Upload(
                user_id=current_user().id,
                original_filename=filename,
                stored_filename=stored_name,
                status="completed",
                header_json=json.dumps(header),
                components_json=json.dumps(components),
                nodes_json=json.dumps(nodes),
                details_json=json.dumps(detail_blocks), 
            )
            db.session.add(upload_row)
            db.session.commit()

            flash("Supply chain map processed successfully.", "success")
            return redirect(url_for("view_upload", upload_id=upload_row.id))

        except Exception as e:
            # Mark as failed
            upload_row = Upload(
                user_id=current_user().id,
                original_filename=filename,
                stored_filename=stored_name,
                status="failed",
            )
            db.session.add(upload_row)
            db.session.commit()
            flash(f"Failed to process file: {e}", "error")
            return redirect(url_for("dashboard"))

    return render_template("upload.html")


@app.route("/upload/<int:upload_id>")
@login_required
def view_upload(upload_id):
    user = current_user()
    upload_row = Upload.query.filter_by(id=upload_id, user_id=user.id).first_or_404()

    header = json.loads(upload_row.header_json) if upload_row.header_json else {}
    components = json.loads(upload_row.components_json) if upload_row.components_json else []
    nodes = json.loads(upload_row.nodes_json) if upload_row.nodes_json else []

    details = json.loads(upload_row.details_json) if upload_row.details_json else []


    return render_template(
        "view_upload.html",
        upload=upload_row,
        header=header,
        components=components,
        nodes=nodes,
        details=details,
    )

@app.route("/upload/<int:upload_id>/delete", methods=["POST"])
@login_required
def delete_upload(upload_id):
    """Delete an uploaded file + its DB record for the current user."""
    user = current_user()
    upload = Upload.query.filter_by(id=upload_id, user_id=user.id).first_or_404()

    # Remove stored file from disk (if it still exists)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], upload.stored_filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        # If file cannot be removed, just ignore and still delete DB row
        pass

    db.session.delete(upload)
    db.session.commit()

    flash("Upload deleted successfully.", "success")
    return redirect(url_for("dashboard"))


# -------------------------
# CLI helper to create DB
# -------------------------

@app.cli.command("init-db")
def init_db():
    """Initialize database tables."""
    db.create_all()
    print("Database initialized.")


if __name__ == "__main__":
    import os
    with app.app_context():
        db.create_all()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


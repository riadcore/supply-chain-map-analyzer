import os
import json
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

    user = db.relationship("User", backref=db.backref("uploads", lazy=True))


# -------------------------
# Auth helpers
# -------------------------

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


def current_user():
    if "user_id" not in session:
        return None
    return User.query.get(session["user_id"])


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


def get_components(df):
    components = []
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
                if isinstance(name, str) and name.strip().startswith("Please list"):
                    break
                if isinstance(name, str) and name.strip():
                    percent = df.iat[k, perc_col] if perc_col < df.shape[1] else None
                    origin = df.iat[k, origin_col] if origin_col < df.shape[1] else None
                    remarks = df.iat[k, remarks_col] if remarks_col < df.shape[1] else None
                    components.append(
                        {
                            "name": name.strip(),
                            "percent": float(percent) if pd.notna(percent) else None,
                            "origin": str(origin).strip() if isinstance(origin, str) else None,
                            "remarks": str(remarks).strip() if isinstance(remarks, str) else None,
                        }
                    )
            break
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
    This is tailored to the sample.xlsx layout.
    """
    nodes = []

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

    for _, col, dg_label in coords_dg:
        material = df.iat[row_material, col]
        company_type = df.iat[row_company_type, col - 1] if col - 1 >= 0 else None

        company_name = df.iat[row_company_name, col + 1] if col + 1 < df.shape[1] else None
        location = df.iat[row_location, col - 1] if col - 1 >= 0 else None

        # If company_name is missing but location cell appears to have both
        if (not isinstance(company_name, str) or not company_name.strip()) and isinstance(location, str):
            company_name = location.strip()
            location = None

        remarks = nearest_value(df, row_remarks, col, window=1)
        date = nearest_value(df, row_date, col, window=1)
        quantity = nearest_value(df, row_qty, col, window=1)

        docs = []
        for r in range(docs_start, docs_end + 1):
            v = df.iat[r, col]
            if isinstance(v, str) and v.strip():
                docs.append(v.strip())

        nodes.append(
            {
                "group": dg_label,
                "material": material.strip() if isinstance(material, str) else None,
                "company_type": company_type.strip() if isinstance(company_type, str) else None,
                "company_name": company_name.strip() if isinstance(company_name, str) else None,
                "location": location.strip() if isinstance(location, str) else None,
                "date": date,
                "quantity": quantity,
                "remarks": remarks,
                "documents": docs,
            }
        )

    return nodes


def parse_supply_chain_excel(file_path: str):
    df = pd.read_excel(file_path, sheet_name=0, header=None)
    header = get_header(df)
    components = get_components(df)
    nodes = get_nodes(df)
    return header, components, nodes


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
@login_required
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
            header, components, nodes = parse_supply_chain_excel(save_path)

            upload_row = Upload(
                user_id=current_user().id,
                original_filename=filename,
                stored_filename=stored_name,
                status="completed",
                header_json=json.dumps(header),
                components_json=json.dumps(components),
                nodes_json=json.dumps(nodes),
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

    return render_template(
        "view_upload.html",
        upload=upload_row,
        header=header,
        components=components,
        nodes=nodes,
    )


# -------------------------
# CLI helper to create DB
# -------------------------

@app.cli.command("init-db")
def init_db():
    """Initialize database tables."""
    db.create_all()
    print("Database initialized.")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

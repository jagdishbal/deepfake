import csv
import io
import logging
import os
import traceback
import uuid
from pathlib import Path

import mysql.connector
from database.database import get_db_connection
from flask import Flask, abort, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from config import MAX_CONTENT_LENGTH, UPLOAD_FOLDER
from services.prediction_service import PredictionError, SUPPORTED_EXTENSIONS, predict_media
from services.audit_service import log_prediction_audit, get_prediction_audit

app = Flask(__name__)
app.secret_key = "deepfake-analysis-secret"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 43200
logger = logging.getLogger(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.after_request
def add_cache_control_headers(response):
    if request.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=43200")
    return response


@app.before_request
def protect_direct_upload_static_access():
    if request.path.startswith("/static/uploads/"):
        abort(404)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/agreement")
def agreement():
    service = request.args.get("service", "image")
    return render_template("agreement.html", service=service)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Email and password are required!", "danger")
            return redirect(url_for("login"))

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        sql = """
        SELECT * FROM users
        WHERE (email=%s OR username=%s) AND password=%s
        """

        cursor.execute(sql, (email, email, password))

        user = cursor.fetchone()

        if not user:
            # Check admin table
            cursor.execute("SELECT * FROM admin WHERE username=%s AND password=%s", (email, password))
            admin_row = cursor.fetchone()
            if admin_row:
                cursor.close()
                connection.close()
                session["user_id"] = admin_row["id"]
                session["username"] = admin_row["username"]
                session["fullname"] = "System Administrator"
                session["admin_name"] = admin_row["username"]
                session["is_admin"] = True
                flash("Admin Login Successful!", "success")
                return redirect(url_for("admin_dashboard"))

        cursor.close()
        connection.close()

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["fullname"] = user["fullname"]
            session["is_admin"] = (user.get("role") == "admin")
            if session["is_admin"]:
                session["admin_name"] = user["username"]

            flash("Login Successful!", "success")
            return redirect(url_for("dashboard"))

        else:
            flash("Invalid Username or Password!", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = (request.form.get("fullname") or "").strip()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        mobile = (request.form.get("mobile") or "").strip()
        password = request.form.get("password") or ""

        if not fullname or not username or not email or not password:
            flash("All required fields must be filled!", "danger")
            return redirect(url_for("register"))

        connection = None
        cursor = None
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)

            # Check whether the username or email already exists
            check_sql = "SELECT id FROM users WHERE username = %s OR email = %s LIMIT 1"
            cursor.execute(check_sql, (username, email))
            existing_user = cursor.fetchone()

            if existing_user:
                flash("Username or Email already registered!", "danger")
                return redirect(url_for("register"))

            sql = """
            INSERT INTO users
            (fullname, username, email, mobile, password)
            VALUES (%s, %s, %s, %s, %s)
            """
            values = (fullname, username, email, mobile, password)
            cursor.execute(sql, values)
            connection.commit()

            flash("Registration Successful!", "success")
            return redirect(url_for("login"))
        except mysql.connector.Error as err:
            if connection:
                connection.rollback()
            if getattr(err, "errno", None) == 1062:  # Duplicate entry key error
                flash("Username or Email already registered!", "danger")
            else:
                logger.exception("Database error during registration: %s", err)
                flash("Registration failed due to a database error. Please try again.", "danger")
            return redirect(url_for("register"))
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

    return render_template("register.html")



@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                COUNT(*) AS total_predictions,
                COALESCE(SUM(CASE WHEN file_type='Image' THEN 1 ELSE 0 END), 0) AS total_images,
                COALESCE(SUM(CASE WHEN file_type='Video' THEN 1 ELSE 0 END), 0) AS total_videos,
                COALESCE(SUM(CASE WHEN file_type='Audio' THEN 1 ELSE 0 END), 0) AS total_audio
            FROM predictions
            WHERE user_id = %s
        """, (session["user_id"],))
        stats = cursor.fetchone() or {}

        total_images = int(stats.get("total_images") or 0)
        total_videos = int(stats.get("total_videos") or 0)
        total_audio = int(stats.get("total_audio") or 0)
        total_predictions = int(stats.get("total_predictions") or 0)

        cursor.execute("""
            SELECT
                id,
                filename AS file_name,
                file_type,
                prediction,
                prediction AS result,
                confidence,
                uploaded_at AS date
            FROM predictions
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            LIMIT 5
        """, (session["user_id"],))
        recent_history = cursor.fetchall()
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    return render_template(
        "dashboard.html",
        total_images=total_images,
        total_videos=total_videos,
        total_audio=total_audio,
        total_predictions=total_predictions,
        accuracy=95,
        history=recent_history,
        session=session
    )

MAX_UPLOAD_BYTES = {
    "image": 10 * 1024 * 1024,
    "video": 100 * 1024 * 1024,
    "audio": 50 * 1024 * 1024,
}

ALLOWED_MIMETYPES = {
    "image": {
        "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
        "image/x-png", "application/octet-stream",
    },
    "video": {
        "video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska", "video/webm",
        "application/octet-stream",
    },
    "audio": {
        "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
        "audio/mpeg", "audio/mp3", "audio/x-mpeg", "audio/x-mp3", "audio/mpeg3", "audio/x-mpeg-3",
        "audio/aac", "audio/x-aac", "audio/aacp",
        "audio/flac", "audio/x-flac",
        "audio/ogg", "application/ogg",
        "audio/m4a", "audio/x-m4a", "audio/mp4", "audio/mp4a-latm",
        "video/mp4",  # MP4 video uploaded for audio extraction
        "application/octet-stream",
    },
}


class UploadValidationError(ValueError):
    pass


def _save_upload(upload, modality: str) -> tuple[str, Path]:
    original_name = secure_filename(upload.filename or "")
    if not original_name:
        raise UploadValidationError("Choose a file with a valid filename.")

    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS[modality]:
        raise UploadValidationError(f"That file format is not supported for {modality} uploads.")

    content_type = (getattr(upload, "mimetype", "") or "").lower().split(";")[0].strip()
    if content_type and content_type not in ALLOWED_MIMETYPES.get(modality, set()):
        if not (modality == "audio" and (content_type.startswith("audio/") or content_type == "video/mp4")):
            raise UploadValidationError(f"That file format or MIME type ({content_type}) is not supported for {modality} uploads.")

    filename = f"{uuid.uuid4().hex}_{original_name}"
    file_path = Path(UPLOAD_FOLDER) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        upload.save(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise UploadValidationError("The uploaded file could not be saved.") from exc

    if file_path.stat().st_size > MAX_UPLOAD_BYTES[modality]:
        file_path.unlink(missing_ok=True)
        limit_mb = MAX_UPLOAD_BYTES[modality] // (1024 * 1024)
        raise UploadValidationError(f"The {modality} file exceeds the {limit_mb} MB limit.")
    if file_path.stat().st_size == 0:
        file_path.unlink(missing_ok=True)
        raise UploadValidationError("The uploaded file is empty.")
    return filename, file_path


def _database_prediction_label(prediction: str) -> str:
    if not isinstance(prediction, str):
        raise ValueError("Prediction label must be a string.")

    normalized = prediction.strip().casefold()
    if normalized == "real":
        return "Real"
    if normalized in {"fake", "deepfake"}:
        return "Deepfake"
    raise ValueError("Prediction label must be Real or Fake.")


def _record_prediction(filename: str, file_type: str, prediction: str, confidence: float) -> int:
    database_prediction = _database_prediction_label(prediction)
    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO predictions
            (user_id, filename, file_type, prediction, confidence)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (session["user_id"], filename, file_type, database_prediction, confidence),
        )
        connection.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("The database did not return the saved prediction ID.")
        return int(cursor.lastrowid)
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def _process_upload(form_key: str, modality: str):
    endpoint = f"{modality}_detection"
    if "user_id" not in session:
        return redirect(url_for("login"))

    upload = request.files.get(form_key)
    if upload is None or not upload.filename:
        flash("Select a file before starting analysis.", "danger")
        return redirect(url_for(endpoint))

    file_path = None
    try:
        filename, file_path = _save_upload(upload, modality)
        result = predict_media(file_path, modality)
        confidence_percent = round(float(result["confidence"]) * 100, 2)
        prediction_id = _record_prediction(
            filename,
            modality.capitalize(),
            result["prediction"],
            confidence_percent,
        )
        log_prediction_audit(
            prediction_id=prediction_id,
            user_id=session["user_id"],
            filename=filename,
            file_type=modality.capitalize(),
            prediction=result["prediction"],
            confidence=confidence_percent,
            model_used=result.get("model_used", f"{modality.capitalize()} Model"),
            model_version=result.get("model_version", "standard"),
            threshold=float(result.get("threshold", 0.50)),
            routing_reason=result.get("routing_reason", "Standard pipeline"),
            probability_real=result.get("probability_real"),
            probability_fake=result.get("probability_fake"),
            is_inconclusive=result.get("is_inconclusive", False),
        )
    except UploadValidationError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(endpoint))
    except PredictionError as exc:
        if file_path is not None:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
        flash(str(exc), "danger")
        return redirect(url_for(endpoint))
    except Exception:
        logger.exception("Media inference or persistence failed for %s upload", modality)
        traceback.print_exc()
        if file_path is not None:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
        flash("The file could not be analyzed right now. Please try again.", "danger")
        return redirect(url_for(endpoint))

    return redirect(url_for("view_result", id=prediction_id))

@app.route("/image_detection", methods=["GET", "POST"])
def image_detection():
    if request.method == "POST":
        return _process_upload("image", "image")
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("upload_image.html")


@app.route("/video_detection", methods=["GET", "POST"])
def video_detection():
    if request.method == "POST":
        return _process_upload("video", "video")
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("upload_video.html")


@app.route("/audio_detection", methods=["GET", "POST"])
def audio_detection():
    if request.method == "POST":
        return _process_upload("audio", "audio")
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("upload_audio.html")


@app.route("/result")
def result():

    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM predictions
        WHERE user_id = %s
        ORDER BY uploaded_at DESC
        LIMIT 1
    """, (session["user_id"],))

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if result is None:
        flash("No prediction found.", "warning")
        return redirect(url_for("dashboard"))

    audit_entry = get_prediction_audit(result["id"]) or {}
    prob_real = audit_entry.get("probability_real")
    prob_fake = audit_entry.get("probability_fake")
    if prob_real is not None and prob_fake is None:
        prob_fake = round(1.0 - prob_real, 4)
    is_inconclusive = bool(audit_entry.get("is_inconclusive", False))

    if prob_real is not None and prob_fake is not None:
        probs_str = f"REAL: {prob_real*100:.1f}%, FAKE: {prob_fake*100:.1f}%"
    else:
        probs_str = None

    conf_str = f"{float(result['confidence']):.2f}%"
    display_prob = probs_str if probs_str else conf_str

    if audit_entry and audit_entry.get("model_used"):
        if is_inconclusive:
            analysis_text = (
                f"INCONCLUSIVE / LOW CONFIDENCE: Classified as {result['prediction']} ({display_prob}), "
                f"but the model score is near the decision threshold or lacks certainty. "
                f"Evaluated with {audit_entry['model_used']} (calibrated threshold: {audit_entry['threshold']:.2f}). "
                f"We recommend additional manual review or a higher-quality media sample."
            )
        else:
            analysis_text = (
                f"Classified as {result['prediction']} ({display_prob}). "
                f"Evaluated with {audit_entry['model_used']} "
                f"(calibrated threshold: {audit_entry['threshold']:.2f}). "
                f"The model score for the selected class is "
                f"{float(result['confidence']):.2f}%."
            )
    else:
        analysis_text = (
            f"Classified as {result['prediction']}. "
            f"The uncalibrated model score for the selected class is "
            f"{float(result['confidence']):.2f}%."
        )

    return render_template(
        "result.html",
        prediction=result["prediction"],
        filename=result["filename"],
        file_type=result["file_type"],
        confidence=float(result["confidence"]),
        analysis=analysis_text,
        preview_file=result["filename"] if (Path(UPLOAD_FOLDER) / result["filename"]).is_file() else None,
        is_inconclusive=is_inconclusive,
        dual_probabilities=probs_str,
    )
@app.route("/result/<int:id>")
def view_result(id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM predictions
        WHERE id=%s AND user_id=%s
    """, (id, session["user_id"]))

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if result is None:
        flash("Prediction not found.", "danger")
        return redirect(url_for("history"))

    audit_entry = get_prediction_audit(result["id"]) or {}
    prob_real = audit_entry.get("probability_real")
    prob_fake = audit_entry.get("probability_fake")
    if prob_real is not None and prob_fake is None:
        prob_fake = round(1.0 - prob_real, 4)
    is_inconclusive = bool(audit_entry.get("is_inconclusive", False))

    if prob_real is not None and prob_fake is not None:
        probs_str = f"REAL: {prob_real*100:.1f}%, FAKE: {prob_fake*100:.1f}%"
    else:
        probs_str = None

    conf_str = f"{float(result['confidence']):.2f}%"
    display_prob = probs_str if probs_str else conf_str

    if audit_entry and audit_entry.get("model_used"):
        if is_inconclusive:
            analysis_text = (
                f"INCONCLUSIVE / LOW CONFIDENCE: Classified as {result['prediction']} ({display_prob}), "
                f"but the model score is near the decision threshold or lacks certainty. "
                f"Evaluated with {audit_entry['model_used']} (calibrated threshold: {audit_entry['threshold']:.2f}). "
                f"We recommend additional manual review or a higher-quality media sample."
            )
        else:
            analysis_text = (
                f"Classified as {result['prediction']} ({display_prob}). "
                f"Evaluated with {audit_entry['model_used']} "
                f"(calibrated threshold: {audit_entry['threshold']:.2f}). "
                f"The model score for the selected class is "
                f"{float(result['confidence']):.2f}%."
            )
    else:
        analysis_text = (
            f"Classified as {result['prediction']}. "
            f"The uncalibrated model score for the selected class is "
            f"{float(result['confidence']):.2f}%."
        )

    return render_template(
        "result.html",
        prediction=result["prediction"],
        filename=result["filename"],
        file_type=result["file_type"],
        confidence=float(result["confidence"]),
        analysis=analysis_text,
        preview_file=result["filename"] if (Path(UPLOAD_FOLDER) / result["filename"]).is_file() else None,
        is_inconclusive=is_inconclusive,
        dual_probabilities=probs_str,
    )
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if Path(filename).name != filename or secure_filename(filename) != filename:
        abort(404)

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id FROM predictions WHERE user_id = %s AND filename = %s",
            (session["user_id"], filename),
        )
        if cursor.fetchone() is None:
            abort(404)
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route("/export_csv")
def export_csv():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT uploaded_at, filename, file_type, prediction, confidence
            FROM predictions
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            """,
            (session["user_id"],),
        )
        rows = cursor.fetchall()
    except Exception:
        logger.exception("Could not read prediction history for CSV export")
        flash("The report could not be generated right now.", "danger")
        return redirect(url_for("history"))
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["Uploaded At", "Filename", "Media Type", "Prediction", "Confidence (%)"])
    writer.writerows(
        (
            row["uploaded_at"],
            row["filename"],
            row["file_type"],
            row["prediction"],
            row["confidence"],
        )
        for row in rows
    )
    csv_file = io.BytesIO(csv_buffer.getvalue().encode("utf-8-sig"))
    return send_file(
        csv_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="deepfake_prediction_report.csv",
    )


@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                id,
                user_id,
                filename,
                filename AS file_name,
                file_type,
                prediction,
                confidence,
                uploaded_at,
                uploaded_at AS date
            FROM predictions
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
        """, (session["user_id"],))
        history_rows = cursor.fetchall()
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    total_images = sum(1 for item in history_rows if item.get("file_type") == "Image")
    total_videos = sum(1 for item in history_rows if item.get("file_type") == "Video")
    total_audio = sum(1 for item in history_rows if item.get("file_type") == "Audio")

    return render_template(
        "history.html",
        history=history_rows,
        total_images=total_images,
        total_videos=total_videos,
        total_audio=total_audio
    )


@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                id,
                fullname,
                username,
                email,
                mobile,
                created_at
            FROM users
            WHERE id = %s
        """, (session["user_id"],))
        user = cursor.fetchone()

        cursor.execute("""
            SELECT 
                COUNT(*) AS total_predictions,
                COALESCE(SUM(CASE WHEN file_type='Image' THEN 1 ELSE 0 END), 0) AS total_images,
                COALESCE(SUM(CASE WHEN file_type='Video' THEN 1 ELSE 0 END), 0) AS total_videos,
                COALESCE(SUM(CASE WHEN file_type='Audio' THEN 1 ELSE 0 END), 0) AS total_audio
            FROM predictions
            WHERE user_id = %s
        """, (session["user_id"],))
        stats = cursor.fetchone() or {}
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    if user is None:
        flash("User not found.", "danger")
        return redirect(url_for("login"))

    total_images = int(stats.get("total_images") or 0)
    total_videos = int(stats.get("total_videos") or 0)
    total_audio = int(stats.get("total_audio") or 0)
    total_predictions = int(stats.get("total_predictions") or 0)

    return render_template(
        "profile.html",
        user=user,
        total_predictions=total_predictions,
        total_images=total_images,
        total_videos=total_videos,
        total_audio=total_audio
    )

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        message = (request.form.get("message") or "").strip()

        if not name or not email or not message:
            flash("Name, email, and message are required fields.", "danger")
            return redirect(url_for("contact"))

        connection = None
        cursor = None
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            sql = """
            INSERT INTO contact (name, email, subject, message)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (name, email, subject or None, message))
            connection.commit()
            flash("Your message has been sent.", "success")
        except Exception as exc:
            if connection:
                connection.rollback()
            logger.exception("Failed to save contact message: %s", exc)
            flash("Could not send your message right now. Please try again.", "danger")
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        rating_str = (request.form.get("rating") or "").strip()
        message = (request.form.get("message") or "").strip()

        if not rating_str or not message:
            flash("Please provide both a rating and a feedback message.", "danger")
            return redirect(url_for("feedback"))

        try:
            rating = int(rating_str)
            if not 1 <= rating <= 5:
                raise ValueError()
        except ValueError:
            flash("Rating must be a number between 1 and 5.", "danger")
            return redirect(url_for("feedback"))

        user_id = session.get("user_id")

        connection = None
        cursor = None
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            sql = """
            INSERT INTO feedback (user_id, rating, message)
            VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (user_id, rating, message))
            connection.commit()
            flash("Feedback submitted successfully.", "success")
        except Exception as exc:
            if connection:
                connection.rollback()
            logger.exception("Failed to save feedback: %s", exc)
            flash("Could not submit feedback right now. Please try again.", "danger")
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
        return redirect(url_for("feedback"))
    return render_template("feedback.html")


@app.route("/admin_dashboard")
def admin_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) AS total_users FROM users")
        total_users = int(cursor.fetchone().get("total_users") or 0)

        cursor.execute("SELECT COUNT(*) AS total_predictions FROM predictions")
        total_predictions = int(cursor.fetchone().get("total_predictions") or 0)

        cursor.execute("""
            SELECT id, fullname, email, created_at 
            FROM users 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_users = cursor.fetchall()
    except Exception as exc:
        logger.exception("Failed to fetch admin stats: %s", exc)
        total_users = 0
        total_predictions = 0
        recent_users = []
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    dataset_size = 3029  # 1699 image + 480 video + 850 audio files
    model_accuracy = 87.9

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        model_accuracy=model_accuracy,
        dataset_size=dataset_size,
        recent_users=recent_users,
        session=session,
    )


@app.route("/users")
def users():
    return redirect(url_for("admin_dashboard"))


@app.route("/datasets")
def datasets():
    return redirect(url_for("admin_dashboard"))


@app.route("/models")
def models():
    return redirect(url_for("admin_dashboard"))


@app.route("/reports")
def reports():
    return redirect(url_for("admin_dashboard"))


@app.route("/settings")
def settings():
    return redirect(url_for("profile"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(413)
def upload_too_large(error):
    flash("The upload exceeds the 100 MB server limit.", "danger")
    if request.path in {"/image_detection", "/video_detection", "/audio_detection"}:
        return redirect(request.path)
    return redirect(url_for("dashboard"))


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html"), 500

if __name__ == "__main__":
    import os
    import threading
    import time
    import webbrowser
    from predict import preload_models

    warmup_thread = threading.Thread(target=preload_models, daemon=True)
    warmup_thread.start()

    def _open_browser():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
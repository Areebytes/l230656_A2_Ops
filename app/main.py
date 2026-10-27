"""
VisionGuard Flask app - Tasks 4 and 5, in four labelled sections:
  [4.4] settings from environment variables
  [T5]  database trail (Prediction table)
  [4.2] upload validation
  [4.1-4.4] app factory: model loaded ONCE, routes, health check, friendly errors

Run from the project root, debug OFF:
  flask --app app.main run --port 8000
  waitress-serve --listen=127.0.0.1:8000 --call app.main:create_app
"""
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

# ============================================== [4.4] settings from env vars
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)   # real env vars win over .env

SRC_DIR = Path(os.getenv("SRC_DIR") or PROJECT_ROOT / "src")   # shared inference.py lives here
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from inference import ImageDecodeError, Predictor  # noqa: E402


def env_path(name, default):
    """Path from an env var; relative values are resolved against the project root."""
    value = Path(os.getenv(name) or default)
    return value if value.is_absolute() else PROJECT_ROOT / value


class Settings:
    def __init__(self):
        self.INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")      # "remote" in Task 8
        self.MODEL_PATH = env_path("MODEL_PATH", "serving_model/model.keras")
        self.MODEL_INFO_PATH = env_path("MODEL_INFO_PATH", "serving_model/model_info.json")
        threshold = os.getenv("THRESHOLD")
        self.THRESHOLD = float(threshold) if threshold else None         # None = champion's own
        default_db = PROJECT_ROOT / "instance" / "visionguard.db"
        self.DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{default_db.as_posix()}"
        self.MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
        self.SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()
        self.ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# ========================================================= [T5] database trail
db = SQLAlchemy()


class Prediction(db.Model):
    """One row per request: what was decided, when, by which model, how confidently."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))          # WHEN (UTC)
    filename = db.Column(db.String(255), nullable=False)                         # WHAT was sent
    image_sha256 = db.Column(db.String(64), nullable=False, index=True)          # (image itself
    file_size = db.Column(db.Integer, nullable=False)                            #  never stored)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    label = db.Column(db.String(20), nullable=False)                             # WHAT was decided
    prob_ai = db.Column(db.Float, nullable=False)                                # HOW confidently
    confidence = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)                       # BY WHICH model
    model_version = db.Column(db.String(50), nullable=False)
    inference_ms = db.Column(db.Float)


# ======================================================= [4.2] upload validation
class UploadError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def validate_upload(file_storage, allowed_extensions):
    """Returns (safe_filename, image_bytes) or raises UploadError with a friendly message.
    Oversized files never get here: Flask rejects them first (MAX_CONTENT_LENGTH -> 413)."""
    if file_storage is None or not file_storage.filename:
        raise UploadError("No file was selected. Please choose an image to upload.")

    original = file_storage.filename
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if ext not in allowed_extensions:
        raise UploadError(f"'.{ext or '?'}' files are not supported. "
                          f"Please upload a JPG or PNG image.", 415)

    # never trust a client-supplied name: strips paths like ..\..\ and odd characters
    safe_name = (secure_filename(original) or f"upload.{ext}")[:255]

    data = file_storage.read()           # in memory only - never written to disk
    if not data:
        raise UploadError("The uploaded file is empty.")

    # right extension is not enough: the CONTENT must be a decodable JPG/PNG
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()                 # structure/integrity check
        with Image.open(BytesIO(data)) as img:
            img.load()                   # full decode (catches truncated files)
            fmt = img.format
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise UploadError("This file looks like an image but could not be decoded. "
                          "It may be corrupted or not a real JPG/PNG.", 422)
    if fmt not in {"JPEG", "PNG"}:
        raise UploadError(f"The file content is {fmt}, not JPG/PNG.", 415)
    return safe_name, data


# ==================================================== app factory and routes
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("visionguard")


def create_app():
    s = Settings()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=s.SECRET_KEY,
        MAX_CONTENT_LENGTH=int(s.MAX_UPLOAD_MB * 1024 * 1024),   # bigger uploads -> 413
        SQLALCHEMY_DATABASE_URI=s.DATABASE_URL,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},        # survive DB restarts
    )

    # [T5] database: create the table if it doesn't exist
    if s.DATABASE_URL.startswith("sqlite:///"):
        Path(s.DATABASE_URL.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    with app.app_context():
        db.create_all()

    # [4.1] load the model ONCE, at start-up (never per request, never from DagsHub)
    if s.INFERENCE_MODE != "local":
        raise RuntimeError(f"INFERENCE_MODE={s.INFERENCE_MODE!r} not supported yet (Task 8)")
    predictor = Predictor(s.MODEL_PATH, s.MODEL_INFO_PATH, s.THRESHOLD)
    log.info("Model %s v%s loaded in %.2fs (threshold %.3f)", predictor.model_name,
             predictor.model_version, predictor.load_seconds, predictor.threshold)

    def show_error(title, message, status):
        """Friendly error: the upload page with an explanation, and the right status code."""
        return render_template("index.html", error_title=title, error=message), status

    @app.context_processor
    def inject_globals():
        return {"model_label": f"{predictor.model_name} v{predictor.model_version}",
                "max_mb": s.MAX_UPLOAD_MB}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        try:
            filename, data = validate_upload(request.files.get("image"), s.ALLOWED_EXTENSIONS)
            result = predictor.predict(data)
        except UploadError as e:
            log.info("Rejected upload: %s", e.message)
            return show_error("Upload rejected", e.message, e.status)
        except ImageDecodeError:
            return show_error("Upload rejected", "The image could not be decoded for analysis.", 422)

        # [T5] permanent record of this decision
        record = Prediction(
            filename=filename, image_sha256=hashlib.sha256(data).hexdigest(),
            file_size=len(data), width=result["width"], height=result["height"],
            label=result["label"], prob_ai=result["prob_ai"], confidence=result["confidence"],
            threshold=result["threshold"], model_name=result["model_name"],
            model_version=result["model_version"], inference_ms=result["inference_ms"])
        db.session.add(record)
        db.session.commit()
        log.info("Prediction #%d: %s (P(AI)=%.3f) by %s v%s", record.id, record.label,
                 record.prob_ai, record.model_name, record.model_version)
        # Post/Redirect/Get: refreshing the result page won't re-submit the upload
        return redirect(url_for("result", prediction_id=record.id))

    @app.get("/result/<int:prediction_id>")
    def result(prediction_id):
        return render_template("result.html", r=db.get_or_404(Prediction, prediction_id))

    @app.get("/history")
    def history():
        rows = db.session.execute(
            db.select(Prediction).order_by(Prediction.created_at.desc()).limit(200)
        ).scalars().all()
        return render_template("history.html", rows=rows)

    @app.get("/health")
    def health():
        try:
            db.session.execute(db.text("SELECT 1"))
            database = "ok"
        except Exception as e:  # report it, don't crash
            log.error("Health check DB failure: %s", e)
            database = "error"
        ok = database == "ok"
        return jsonify(status="ok" if ok else "degraded", model_loaded=True,
                       model_name=predictor.model_name, model_version=predictor.model_version,
                       threshold=round(predictor.threshold, 4), img_size=predictor.img_size,
                       resize_mode=predictor.resize_mode, database=database), (200 if ok else 503)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        return show_error("File too large", f"Images must be at most {s.MAX_UPLOAD_MB:g} MB.", 413)

    @app.errorhandler(HTTPException)
    def http_error(e):
        return show_error(f"{e.code} {e.name}", e.description, e.code)

    @app.errorhandler(Exception)
    def unexpected(_e):
        log.exception("Unhandled error")
        return show_error("Something went wrong", "An unexpected error occurred. Please try again.", 500)

    return app
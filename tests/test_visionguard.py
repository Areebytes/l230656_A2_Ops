"""
VisionGuard tests (Task 6.6). Run with: pytest -q
A tiny dummy model is built on the fly, so CI needs neither the real
model file nor DagsHub access.
"""
import io
import json

import keras
import pytest
import tensorflow as tf
from PIL import Image

from inference import ImageDecodeError, Predictor, resize_image


def png_bytes(width=300, height=200):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Tiny stand-in model with the same input/output shape as the champion."""
    d = tmp_path_factory.mktemp("model")
    inp = keras.Input((160, 160, 3))
    x = keras.layers.GlobalAveragePooling2D()(inp)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    keras.Model(inp, out).save(d / "model.keras")
    (d / "model_info.json").write_text(json.dumps({
        "model_name": "test-model", "version": "0", "threshold": 0.5,
        "img_size": 160, "resize_mode": "crop"}))
    return d


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    """Flask test client using the dummy model and a throwaway SQLite database."""
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.keras"))
    monkeypatch.setenv("MODEL_INFO_PATH", str(model_dir / "model_info.json"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("THRESHOLD", "")
    from app.main import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------- preprocessing
@pytest.mark.parametrize("mode", ["crop", "pad", "stretch"])
def test_resize_gives_model_input_shape(mode):
    image = tf.zeros((300, 200, 3), tf.uint8)
    out = resize_image(image, 160, mode)
    assert out.shape == (160, 160, 3)
    assert out.dtype == tf.uint8


# -------------------------------------------------------------------- predictor
def test_predictor_returns_label_and_confidence(model_dir):
    p = Predictor(model_dir / "model.keras", model_dir / "model_info.json")
    r = p.predict(png_bytes())
    assert r["label"] in {"AI-generated", "Real"}
    assert 0.0 <= r["prob_ai"] <= 1.0
    assert r["confidence"] >= 0.5
    assert r["model_version"] == "0"


def test_predictor_rejects_non_image(model_dir):
    p = Predictor(model_dir / "model.keras", model_dir / "model_info.json")
    with pytest.raises(ImageDecodeError):
        p.predict(b"this is not an image")


# --------------------------------------------------------------------- web app
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_valid_upload_is_saved_to_history(client):
    r = client.post("/predict", data={"image": (io.BytesIO(png_bytes()), "photo.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 302                          # redirect to the result page
    assert b"photo.png" in client.get("/history").data   # recorded in the database


def test_wrong_file_type_rejected(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"hello"), "notes.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 415


def test_fake_image_rejected(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "fake.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 422


def test_filename_is_sanitized(client):
    client.post("/predict", data={"image": (io.BytesIO(png_bytes()), "../../evil name.png")},
                content_type="multipart/form-data")
    history = client.get("/history").data
    assert b"evil_name.png" in history
    assert b"../" not in history
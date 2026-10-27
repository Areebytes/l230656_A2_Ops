"""
VisionGuard inference - shared by training, the Flask app (Task 4) and the cloud API (Task 7).

  1. decode_image / resize_image : the ONE preprocessing implementation
                                   (data_pipeline.py imports these for training)
  2. Predictor                   : loads the model ONCE, classifies image bytes
  3. CLI
       python src/inference.py export              -> download @champion ONCE into serving_model/
       python src/inference.py predict <img> ...   -> console check (Task 4.1)

Top-level imports are TensorFlow/Keras/NumPy only; MLflow is imported only by `export`,
so the serving app never needs MLflow or DagsHub credentials.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


# =========================================================== 1. preprocessing
def decode_image(raw_bytes):
    """Bytes -> uint8 tensor (H, W, 3). channels=3 converts L / P / RGBA to RGB."""
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return image


def resize_image(image, size, mode):
    """uint8 (H, W, 3) -> uint8 (size, size, 3).

    crop    : centre square crop, then resize. Keeps proportions, loses edges.
    pad     : resize to fit, fill the rest with black bars. Keeps everything.
    stretch : squash to a square. Keeps everything, distorts shapes.
    """
    if mode == "crop":
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        side = tf.minimum(h, w)
        image = tf.image.crop_to_bounding_box(image, (h - side) // 2, (w - side) // 2, side, side)
        image = tf.image.resize(image, [size, size], antialias=True)
    elif mode == "pad":
        image = tf.image.resize_with_pad(image, size, size, antialias=True)
    elif mode == "stretch":
        image = tf.image.resize(image, [size, size], antialias=True)
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


# =============================================================== 2. predictor
class ImageDecodeError(ValueError):
    """Bytes could not be decoded as an image by TensorFlow."""


class Predictor:
    def __init__(self, model_path, info_path=None, threshold=None):
        start = time.perf_counter()
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path} "
                                    f"(run 'python src/inference.py export' or set MODEL_PATH)")
        self.model = keras.models.load_model(model_path, compile=False)

        info = {}
        if info_path and Path(info_path).exists():
            info = json.loads(Path(info_path).read_text())
        self.info = info
        self.model_name = info.get("model_name", model_path.stem)
        self.model_version = str(info.get("version", "unknown"))
        self.img_size = int(info.get("img_size", self.model.input_shape[1]))
        self.resize_mode = info.get("resize_mode", "crop")
        # THRESHOLD env var (if given) overrides the champion's validated threshold
        self.threshold = float(threshold if threshold is not None else info.get("threshold", 0.5))
        if self.model.input_shape[1] != self.img_size:
            raise ValueError("model input size does not match model_info.json img_size")

        # Warm-up: the first call builds TensorFlow's graph (slow). Pay that cost
        # at start-up, not on the first user's request.
        self.model(np.zeros((1, self.img_size, self.img_size, 3), np.float32), training=False)
        self.load_seconds = time.perf_counter() - start

    def predict(self, image_bytes: bytes) -> dict:
        start = time.perf_counter()
        try:
            image = decode_image(tf.constant(image_bytes))
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            raise ImageDecodeError("image could not be decoded") from e
        height, width = int(image.shape[0]), int(image.shape[1])

        # EXACTLY the training preprocessing; pixel scaling happens inside the model
        x = tf.cast(resize_image(image, self.img_size, self.resize_mode)[None, ...], tf.float32)
        prob_ai = float(self.model(x, training=False)[0, 0])

        is_ai = prob_ai >= self.threshold
        return {
            "label": "AI-generated" if is_ai else "Real",
            "prob_ai": prob_ai,
            "confidence": prob_ai if is_ai else 1.0 - prob_ai,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "width": width,
            "height": height,
            "inference_ms": (time.perf_counter() - start) * 1000,
        }


# ===================================================================== 3. CLI
def export_champion(out_dir: Path):
    """Registry (DagsHub) -> out_dir/model.keras + model_info.json. Run ONCE."""
    import hashlib
    import shutil
    import tempfile
    from datetime import datetime, timezone

    import mlflow
    from mlflow import MlflowClient

    from tracking import REGISTERED_MODEL, setup_mlflow

    setup_mlflow()
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion")
    print(f"Champion = version {mv.version} (run {mv.tags.get('run_name')}, run_id {mv.run_id})")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    with tempfile.TemporaryDirectory() as tmp:
        # the exact best.keras the run produced (logged in 3.4 as keras_model/best.keras)
        local = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="keras_model/best.keras", dst_path=tmp)
        shutil.copy(local, model_path)

    info = {
        "model_name": REGISTERED_MODEL,
        "version": mv.version,
        "alias": "champion",
        "run_id": mv.run_id,
        "run_name": mv.tags.get("run_name"),
        "threshold": float(mv.tags["threshold"]),
        "img_size": int(mv.tags["img_size"]),
        "resize_mode": mv.tags["resize_mode"],
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "tensorflow_version": tf.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))

    # Verify before declaring success: loads, sizes match, warm-up prediction works
    Predictor(model_path, out_dir / "model_info.json")
    print(json.dumps(info, indent=2))
    print(f"\nExported to {out_dir}  ({model_path.stat().st_size / 1e6:.1f} MB)")


def main():
    import argparse

    from config import PROJECT_ROOT   # loads .env (MLflow credentials for export)

    model_dir = PROJECT_ROOT / "serving_model"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="download the @champion model once")
    pred = sub.add_parser("predict", help="classify images with the exported model")
    pred.add_argument("images", nargs="+")
    args = parser.parse_args()

    if args.command == "export":
        export_champion(model_dir)
        return

    p = Predictor(model_dir / "model.keras", model_dir / "model_info.json")
    print(f"Loaded {p.model_name} v{p.model_version} ONCE in {p.load_seconds:.2f}s "
          f"(threshold={p.threshold:.3f}, {p.img_size}px, {p.resize_mode})")
    for raw in args.images:
        path = Path(raw) if Path(raw).exists() else PROJECT_ROOT / raw
        r = p.predict(path.read_bytes())
        print(f"{path.name}: {r['label']}  P(AI)={r['prob_ai']:.4f}  "
              f"confidence={r['confidence']:.1%}  ({r['inference_ms']:.0f} ms)")


if __name__ == "__main__":
    main()
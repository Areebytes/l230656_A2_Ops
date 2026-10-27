"""
VisionGuard - Task 3.6: load the model by alias and classify one image.
Usage: python src/predict_champion.py <image path>
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import mlflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from data_pipeline import decode_image, resize_image  # noqa: E402
from tracking import REGISTERED_MODEL  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    uri = f"models:/{REGISTERED_MODEL}@champion"
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion")
    threshold = float(mv.tags.get("threshold", 0.5))
    img_size = int(mv.tags.get("img_size", 160))
    resize_mode = mv.tags.get("resize_mode", "crop")
    print(f"Loading {uri} -> version {mv.version} (run {mv.tags.get('run_name')})")
    model = mlflow.tensorflow.load_model(uri)

    # Same preprocessing as training: decode -> RGB -> resize (normalization is inside the model)
    image = resize_image(decode_image(tf.io.read_file(args.image)), img_size, resize_mode)
    prob_ai = float(model.predict(tf.cast(image[None, ...], tf.float32), verbose=0)[0, 0])

    label = "AI-generated" if prob_ai >= threshold else "Real"
    confidence = prob_ai if label == "AI-generated" else 1 - prob_ai
    print(f"Image:      {args.image}")
    print(f"P(AI):      {prob_ai:.4f}   threshold: {threshold:.4f}   "
          f"(img_size={img_size}, resize={resize_mode})")
    print(f"Prediction: {label}  (confidence {confidence:.1%})")


if __name__ == "__main__":
    main()  
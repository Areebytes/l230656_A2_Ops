"""
VisionGuard - model definitions.

Every model takes raw float32 pixels in [0, 255] and normalizes INSIDE the
model, so the saved .keras file carries its own preprocessing.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from config import IMG_SIZE  # noqa: E402


def conv_block(x, filters, name):
    """Conv -> BatchNorm -> ReLU -> MaxPool."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_baseline_cnn(img_size=IMG_SIZE, dropout=0.4):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = conv_block(x, filters, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def layer_table(model):
    """Per-layer output shape and parameter count (evidence for 2.2)."""
    rows = [{"layer": l.name,
             "type": l.__class__.__name__,
             "output_shape": str(tuple(l.output.shape)),
             "params": l.count_params()} for l in model.layers]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from config import REPORTS_DIR

    model = build_baseline_cnn()
    model.summary()
    table = layer_table(model)
    print("\n" + table.to_string(index=False))
    out = REPORTS_DIR / "baseline_cnn_layers.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved {out}")
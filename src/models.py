"""
VisionGuard - model definitions.

Every model takes raw float32 pixels in [0, 255] and normalizes INSIDE the
model, so the saved .keras file carries its own preprocessing.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from config import IMG_SIZE  # noqa: E402

# ---------------------------------------------------------------- baseline CNN


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


# ---------------------------------------------------------- transfer learning

# First layer of MobileNetV2's final inverted-residual block
LAST_BLOCK_START = "block_16_expand"


def build_transfer_model(img_size=IMG_SIZE, dropout=0.3):
    """MobileNetV2 (ImageNet weights, no classifier) + our own head.
    Returns (model, backbone) so the training script can unfreeze layers later."""
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False  # Stage 1: freeze everything

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    # Same maths as mobilenet_v2.preprocess_input: [0, 255] -> [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    # training=False keeps BatchNorm in inference mode (ImageNet statistics), even when fine-tuning
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer"), base


def layers_in_last_block(base):
    """How many backbone layers belong to the last block (block_16 + final 1x1 conv)."""
    names = [l.name for l in base.layers]
    return len(names) - names.index(LAST_BLOCK_START)


def unfreeze_top(base, n_layers):
    """Unfreeze the top n backbone layers; BatchNorm layers stay frozen."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


# ------------------------------------------------------------------ evidence


def count_params(weights):
    return int(sum(np.prod(w.shape) for w in weights))


def trainable_summary(model, base, title):
    """Which layers train in this stage (evidence for 2.3)."""
    backbone = [l.name for l in base.layers if l.trainable and l.weights]
    head = [l.name for l in model.layers if l is not base and l.trainable and l.weights]
    lines = [f"=== {title} ===",
             f"Trainable params:     {count_params(model.trainable_weights):,}",
             f"Non-trainable params: {count_params(model.non_trainable_weights):,}",
             f"Trainable backbone layers ({len(backbone)} of {len(base.layers)}):"]
    lines += [f"  {n}" for n in backbone] or ["  (none - backbone frozen)"]
    lines.append("Trainable head layers:")
    lines += [f"  {n}" for n in head]
    return "\n".join(lines)


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
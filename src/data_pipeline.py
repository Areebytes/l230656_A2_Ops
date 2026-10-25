"""
VisionGuard - Task 2.1: tf.data input pipeline.

Stages:  read split CSV -> (path, label) -> decode -> resize -> cache
         -> [train only: shuffle + augment] -> float32 -> batch -> prefetch

Images leave the pipeline as float32 in [0, 255]. Each MODEL normalizes
them itself (first layer), so serving only has to decode + resize.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import (BATCH_SIZE, DATA_DIR, IMG_SIZE, LABEL_TO_CLASS,  # noqa: E402
                    RESIZE_MODE, SEED, SPLITS_DIR)

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Shared preprocessing: Flask/API inference will import THESE functions,
# so training and serving can never drift apart.
# ---------------------------------------------------------------------------
def decode_image(raw_bytes):
    """Bytes -> uint8 tensor (H, W, 3). channels=3 converts L / P / RGBA to RGB."""
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return image


def resize_image(image, size=IMG_SIZE, mode=RESIZE_MODE):
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
        raise ValueError(f"Unknown RESIZE_MODE: {mode}")
    # back to uint8: 4x less memory in the cache than float32
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def load_split(name):
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    paths = [str(DATA_DIR / p) for p in df["path"]]
    labels = df["label"].astype("float32").to_numpy()  # float for binary cross-entropy
    return paths, labels


def load_and_resize(path, label):
    return resize_image(decode_image(tf.io.read_file(path))), label


def augment(image, label):
    """Random, label-preserving changes. Training only."""
    image = tf.image.random_flip_left_right(image)
    # Re-compress at random JPEG quality so compression artifacts stop being a
    # class shortcut (audit: AI class has far more PNGs than the real class).
    image = tf.image.random_jpeg_quality(image, 60, 100)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=20.0)  # in 0-255 units
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0), label


def to_float(image, label):
    return tf.cast(image, tf.float32), label


def make_dataset(split, training=False, batch_size=BATCH_SIZE, use_augmentation=True):
    paths, labels = load_split(split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_resize, num_parallel_calls=AUTOTUNE)  # parallel decode
    ds = ds.cache()                                             # decode big JPEGs once
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment if use_augmentation else to_float, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(to_float, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)              # overlap CPU prep & training


# ---------------------------------------------------------------------------
# Evidence: dataset info + figure of one augmented training batch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config import FIGURES_DIR

    print(f"IMG_SIZE={IMG_SIZE}  RESIZE_MODE={RESIZE_MODE}  BATCH_SIZE={BATCH_SIZE}")
    for split in ["train", "val", "test"]:
        n = len(load_split(split)[0])
        print(f"{split:5s}: {n} images -> {-(-n // BATCH_SIZE)} batches")

    train_ds = make_dataset("train", training=True)
    print("element_spec:", train_ds.element_spec)

    images, labels = next(iter(train_ds))
    n_show = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(LABEL_TO_CLASS[int(labels[i])], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch ({IMG_SIZE}px, {RESIZE_MODE})")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "augmented_batch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
"""
VisionGuard - training script.

Fixed seeds, early stopping, checkpoint on val_loss, LR reduction (Task 2.4).
Usage:  python src/train.py --model cnn
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGURES_DIR, MODELS_DIR, SEED  # noqa: E402
from data_pipeline import make_dataset  # noqa: E402
from models import build_baseline_cnn  # noqa: E402

METRICS = ["accuracy", "auc", "precision", "recall"]


def set_seeds(seed=SEED):
    """Seeds Python's random, NumPy and TensorFlow in one call."""
    keras.utils.set_random_seed(seed)


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )


def make_callbacks(run_dir, patience):
    return [
        # stop when val_loss hasn't improved for `patience` epochs; roll back to best weights
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        # always keep the best model on disk (crash-safe, and it's what we deploy)
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        # halve the LR when val_loss stalls -> finer steps near a minimum
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, patience // 3),
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]


def plot_history(hist, best_epoch, title, out_path, stage_boundary=None):
    epochs = np.arange(1, len(hist) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, hist[metric], label=f"train {metric}")
        ax.plot(epochs, hist[f"val_{metric}"], label=f"val {metric}")
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch ({best_epoch})")
        if stage_boundary:  # used by transfer learning (2.3)
            ax.axvline(stage_boundary + 0.5, color="gray", linestyle=":", label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def report_best(hist, label=""):
    best_idx = int(hist["val_loss"].idxmin())
    best = hist.iloc[best_idx]
    print(f"\n{label}Best epoch: {best_idx + 1}")
    for m in ["loss"] + METRICS:
        print(f"  train {m:9s} {best[m]:.4f}   val {m:9s} {best['val_' + m]:.4f}")
    return best_idx + 1


def train_cnn(args, train_ds, val_ds):
    run_dir = MODELS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_baseline_cnn(dropout=args.dropout)
    compile_model(model, args.lr)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=make_callbacks(run_dir, args.patience))

    hist = pd.DataFrame(history.history)
    best_epoch = report_best(hist)
    plot_history(hist, best_epoch, f"Baseline CNN ({args.run_name})",
                 FIGURES_DIR / f"{args.run_name}_curves.png")
    print(f"Best model saved to {run_dir / 'best.keras'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn"], default="cnn")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    args.run_name = args.run_name or f"{args.model}_baseline"

    set_seeds()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = make_dataset("train", training=True, use_augmentation=not args.no_augment)
    val_ds = make_dataset("val")

    if args.model == "cnn":
        train_cnn(args, train_ds, val_ds)


if __name__ == "__main__":
    main()
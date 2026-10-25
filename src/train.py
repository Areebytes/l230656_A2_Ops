"""
VisionGuard - training script.

Fixed seeds, early stopping, checkpoint on val_loss, LR reduction (Task 2.4).
Usage:
  python src/train.py --model cnn
  python src/train.py --model transfer
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

from config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED  # noqa: E402
from data_pipeline import make_dataset  # noqa: E402
from models import (build_baseline_cnn, build_transfer_model,  # noqa: E402
                    layers_in_last_block, trainable_summary, unfreeze_top)

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


def make_callbacks(run_dir, patience, csv_name="history.csv",
                   initial_threshold=None, min_lr=1e-6):
    return [
        # stop when val_loss hasn't improved for `patience` epochs; roll back to best weights
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        # keep the best model on disk; initial_value_threshold lets stage 2 overwrite
        # the file ONLY if it beats stage 1's best val_loss
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_loss",
                                        save_best_only=True,
                                        initial_value_threshold=initial_threshold,
                                        verbose=1),
        # halve the LR when val_loss stalls -> finer steps near a minimum
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, patience // 3),
                                          min_lr=min_lr, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir / csv_name)),
    ]


def plot_history(hist, best_epoch, title, out_path, first_epoch=1, stage_boundary=None):
    epochs = np.arange(first_epoch, first_epoch + len(hist))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, hist[metric], label=f"train {metric}")
        ax.plot(epochs, hist[f"val_{metric}"], label=f"val {metric}")
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch ({best_epoch})")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="gray", linestyle=":",
                       label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def report_best(hist, label="", first_epoch=1):
    best_idx = int(hist["val_loss"].to_numpy().argmin())
    best = hist.iloc[best_idx]
    epoch = first_epoch + best_idx
    print(f"\n{label}Best epoch: {epoch}")
    for m in ["loss"] + METRICS:
        print(f"  train {m:9s} {best[m]:.4f}   val {m:9s} {best['val_' + m]:.4f}")
    return epoch


def save_text(text, path):
    path.write_text(text, encoding="utf-8")
    print(f"Saved {path}")


# --------------------------------------------------------------- CNN (2.2)


def train_cnn(args, train_ds, val_ds, run_dir):
    model = build_baseline_cnn(dropout=args.dropout)
    compile_model(model, args.lr)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=make_callbacks(run_dir, args.patience))
    hist = pd.DataFrame(history.history)
    best_epoch = report_best(hist)
    plot_history(hist, best_epoch, f"Baseline CNN ({args.run_name})",
                 FIGURES_DIR / f"{args.run_name}_curves.png")


# --------------------------------------------------- transfer learning (2.3)


def train_transfer(args, train_ds, val_ds, run_dir):
    model, base = build_transfer_model(dropout=args.dropout)

    # ---- Stage 1: backbone frozen, train only the new head ----
    compile_model(model, args.lr)
    s1 = trainable_summary(model, base, "Stage 1 (backbone frozen)")
    print(s1)
    save_text(s1, REPORTS_DIR / f"{args.run_name}_stage1_trainable.txt")

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.stage1_epochs,
                   callbacks=make_callbacks(run_dir, args.patience, "history_stage1.csv"))
    hist1 = pd.DataFrame(h1.history)
    best1 = report_best(hist1, "[Stage 1] ")
    plot_history(hist1, best1, f"MobileNetV2 stage 1: frozen backbone ({args.run_name})",
                 FIGURES_DIR / f"{args.run_name}_stage1_curves.png")

    # ---- Stage 2: unfreeze the last block, ~100x smaller LR ----
    # EarlyStopping restored stage 1's best weights, so fine-tuning starts from the best point.
    n_layers = args.unfreeze_layers or layers_in_last_block(base)
    unfreeze_top(base, n_layers)
    compile_model(model, args.ft_lr)  # trainable changes only take effect after re-compiling
    s2 = trainable_summary(model, base, f"Stage 2 (top {n_layers} backbone layers unfrozen)")
    print(s2)
    save_text(s2, REPORTS_DIR / f"{args.run_name}_stage2_trainable.txt")

    done = len(hist1)  # epochs actually run in stage 1 (early stopping may cut it short)
    h2 = model.fit(train_ds, validation_data=val_ds,
                   initial_epoch=done, epochs=done + args.stage2_epochs,
                   callbacks=make_callbacks(run_dir, args.patience, "history_stage2.csv",
                                            initial_threshold=float(hist1["val_loss"].min()),
                                            min_lr=1e-7))
    hist2 = pd.DataFrame(h2.history)
    best2 = report_best(hist2, "[Stage 2] ", first_epoch=done + 1)
    plot_history(hist2, best2, f"MobileNetV2 stage 2: fine-tuning ({args.run_name})",
                 FIGURES_DIR / f"{args.run_name}_stage2_curves.png", first_epoch=done + 1)

    full = pd.concat([hist1, hist2], ignore_index=True)
    best_all = report_best(full, "[Overall] ")
    plot_history(full, best_all, f"MobileNetV2 both stages ({args.run_name})",
                 FIGURES_DIR / f"{args.run_name}_all_curves.png", stage_boundary=done)


# --------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR (stage 1 for transfer)")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-augment", action="store_true")
    # CNN
    parser.add_argument("--epochs", type=int, default=40)
    # transfer
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--ft-lr", type=float, default=1e-5, help="fine-tuning LR")
    parser.add_argument("--unfreeze-layers", type=int, default=None,
                        help="backbone layers to unfreeze in stage 2 (default: last block)")
    args = parser.parse_args()

    defaults = {"cnn": ("cnn_baseline", 0.4), "transfer": ("transfer_mobilenetv2", 0.3)}
    default_name, default_dropout = defaults[args.model]
    args.run_name = args.run_name or default_name
    args.dropout = default_dropout if args.dropout is None else args.dropout

    set_seeds()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = MODELS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset("train", training=True, use_augmentation=not args.no_augment)
    val_ds = make_dataset("val")

    if args.model == "cnn":
        train_cnn(args, train_ds, val_ds, run_dir)
    else:
        train_transfer(args, train_ds, val_ds, run_dir)
    print(f"\nBest model: {run_dir / 'best.keras'}")


if __name__ == "__main__":
    main()
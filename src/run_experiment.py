"""
VisionGuard - one tracked experiment run on DagsHub MLflow.

Task 3.3: params, per-epoch metrics, test metrics, tags; manual OR autolog
Task 3.4: artifacts - confusion matrix, ROC, curves, split lists, packages, Keras model
Task 3.5: called once per experiment (see run_experiments.ps1)

Image size and resize mode come from env vars IMG_SIZE / RESIZE_MODE (config.py).
"""
import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import keras  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from config import BATCH_SIZE, IMG_SIZE, MODELS_DIR, RESIZE_MODE, SEED, SPLITS_DIR  # noqa: E402
from data_pipeline import make_dataset  # noqa: E402
from evaluate import choose_threshold, compute_metrics, plot_confusion, predict  # noqa: E402
from models import (build_baseline_cnn, build_transfer_model,  # noqa: E402
                    layers_from, unfreeze_top)
from tracking import (MLflowEpochLogger, common_tags, log_environment,  # noqa: E402
                      setup_mlflow)
from train import compile_model, make_callbacks, plot_history, set_seeds  # noqa: E402


# ------------------------------------------------------------------ training

def train_cnn(args, train_ds, val_ds, run_dir, extra_callbacks):
    model = build_baseline_cnn(dropout=args.dropout)
    compile_model(model, args.lr)
    h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                  callbacks=make_callbacks(run_dir, args.patience) + extra_callbacks)
    return pd.DataFrame(h.history), None


def train_transfer(args, train_ds, val_ds, run_dir, extra_callbacks):
    model, base = build_transfer_model(dropout=args.dropout)
    compile_model(model, args.lr)                                   # stage 1: frozen
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.stage1_epochs,
                   callbacks=make_callbacks(run_dir, args.patience, "history_stage1.csv")
                   + extra_callbacks)
    hist1 = pd.DataFrame(h1.history)

    unfreeze_top(base, layers_from(base, f"block_{17 - args.unfreeze_blocks}_expand"))
    compile_model(model, args.ft_lr)                                # stage 2: fine-tune
    done = len(hist1)
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=done,
                   epochs=done + args.stage2_epochs,
                   callbacks=make_callbacks(run_dir, args.patience, "history_stage2.csv",
                                            initial_threshold=float(hist1["val_loss"].min()),
                                            min_lr=1e-7) + extra_callbacks)
    return pd.concat([hist1, pd.DataFrame(h2.history)], ignore_index=True), done


# ------------------------------------------------------------------ plotting

def plot_roc(y, p, m, title, out):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {m['roc_auc']:.3f}")
    ax.scatter(m["fpr_real_called_ai"], m["recall"], color="red", zorder=5,
               label=f"operating point (t={m['threshold']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR (real called AI)")
    ax.set_ylabel("TPR (AI caught)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "transfer"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
    p.add_argument("--description", default="")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--epochs", type=int, default=40)                     # cnn
    p.add_argument("--stage1-epochs", type=int, default=15)              # transfer
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-blocks", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--threshold-strategy", default="max_fpr")
    p.add_argument("--max-fpr", type=float, default=0.10)
    args = p.parse_args()
    args.dropout = args.dropout if args.dropout is not None else (0.4 if args.model == "cnn" else 0.3)

    if args.tracking == "autolog" and args.model == "transfer":
        raise SystemExit("Autolog assumes one fit() per run; two-stage transfer learning would "
                         "re-log params with new values. Use --tracking manual for transfer.")

    setup_mlflow()
    set_seeds()
    run_dir = MODELS_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_ds = make_dataset("train", training=True, use_augmentation=not args.no_augment)
    val_ds = make_dataset("val")

    # ---- 3.3: hyperparameters (manual mode) ----
    params = {"model": args.model, "img_size": IMG_SIZE, "resize_mode": RESIZE_MODE,
              "batch_size": BATCH_SIZE, "augmentation": not args.no_augment, "lr": args.lr,
              "dropout": args.dropout, "patience": args.patience, "seed": SEED,
              "threshold_strategy": args.threshold_strategy, "max_fpr": args.max_fpr}
    if args.model == "cnn":
        params["epochs"] = args.epochs
    else:
        params.update({"stage1_epochs": args.stage1_epochs, "stage2_epochs": args.stage2_epochs,
                       "ft_lr": args.ft_lr, "unfreeze_blocks": args.unfreeze_blocks,
                       "backbone": "MobileNetV2"})

    with mlflow.start_run(run_name=args.run_name, description=args.description) as run:
        # ---- 3.3: tags ----
        mlflow.set_tags({**common_tags(), "model_type": args.model,
                         "tracking_method": args.tracking})

        if args.tracking == "autolog":
            # Autolog patches fit(): logs fit/optimizer params, per-epoch metrics,
            # early-stopping info, model summary and the model. We log NO params ourselves.
            mlflow.tensorflow.autolog(log_models=True)
            extra = []
        else:
            mlflow.log_params(params)
            extra = [MLflowEpochLogger()]

        train_fn = train_cnn if args.model == "cnn" else train_transfer
        hist, boundary = train_fn(args, train_ds, val_ds, run_dir, extra)
        best_idx = int(hist["val_loss"].to_numpy().argmin())
        mlflow.log_metrics({"best_epoch": best_idx + 1,
                            "best_val_loss": float(hist["val_loss"].iloc[best_idx]),
                            "best_val_auc": float(hist["val_auc"].iloc[best_idx])})

        # ---- evaluate the BEST checkpoint: threshold on val, metrics on test ----
        model = keras.models.load_model(run_dir / "best.keras", compile=False)
        val = predict(model, "val")
        yv, pv = val["label"].to_numpy(), val["prob_ai"].to_numpy()
        t = choose_threshold(yv, pv, args.threshold_strategy, args.max_fpr)
        val_m = compute_metrics(yv, pv, t)
        test = predict(model, "test")
        yt, pt = test["label"].to_numpy(), test["prob_ai"].to_numpy()
        test_m = compute_metrics(yt, pt, t)

        # ---- 3.3: final metrics (prefixed so they don't clash with per-epoch curves) ----
        mlflow.log_metric("decision_threshold", t)
        for prefix, m in (("final_val", val_m), ("test", test_m)):
            mlflow.log_metrics({f"{prefix}_{k}": float(v) for k, v in m.items()
                                if k != "threshold"})

        # ---- 3.4: artifacts ----
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_history(hist, best_idx + 1, args.run_name, tmp / "training_curves.png",
                         stage_boundary=boundary)
            plot_confusion(yt, pt, t, f"{args.run_name} - test", tmp / "confusion_matrix_test.png")
            plot_roc(yt, pt, test_m, f"{args.run_name} - ROC (test)", tmp / "roc_curve_test.png")
            mlflow.log_artifacts(str(tmp), "plots")
        mlflow.log_artifacts(str(SPLITS_DIR), "splits")
        log_environment()
        mlflow.log_artifact(str(run_dir / "best.keras"), "keras_model")    # raw file, for serving

        if args.tracking == "manual":
            info = mlflow.tensorflow.log_model(
                model, name="model",
                input_example=np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
            mlflow.set_tag("model_uri", info.model_uri)                    # used by 3.6

        print(f"\nRun {run.info.run_id}  best_epoch={best_idx + 1}  "
              f"val_auc={val_m['roc_auc']:.3f}  test_auc={test_m['roc_auc']:.3f}  t={t:.3f}")


if __name__ == "__main__":
    main()
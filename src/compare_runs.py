"""
VisionGuard - Task 3.5: compare all runs in one table and chart, choose the best.
Selection uses VALIDATION metrics only; test metrics are shown but never used to choose.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from tracking import EXPERIMENT_NAME, setup_mlflow  # noqa: E402

COLUMNS = {
    "tags.mlflow.runName": "run",
    "tags.tracking_method": "tracking",
    "params.model": "model",
    "params.img_size": "img",
    "params.resize_mode": "resize",
    "params.augmentation": "aug",
    "params.ft_lr": "ft_lr",
    "params.unfreeze_blocks": "unfreeze",
    "metrics.best_epoch": "best_epoch",
    "metrics.best_val_loss": "val_loss",
    "metrics.final_val_roc_auc": "val_auc",
    "metrics.final_val_recall": "val_recall@FPR<=10%",
    "metrics.final_val_fpr_real_called_ai": "val_fpr",
    "metrics.test_roc_auc": "test_auc",
    "metrics.test_accuracy": "test_acc",
    "metrics.test_f1": "test_f1",
}


def main():
    setup_mlflow()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                              filter_string="attributes.status = 'FINISHED'")
    runs = runs[[c for c in COLUMNS if c in runs.columns]].rename(columns=COLUMNS)
    runs = runs.sort_values("val_auc", ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(runs.to_string(index=False))
    runs.to_csv(REPORTS_DIR / "experiment_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = runs.sort_values("val_auc")
    axes[0].barh(order["run"], order["val_auc"], color="tab:blue")
    axes[0].set_xlabel("validation ROC-AUC (higher = better)")
    axes[0].set_xlim(0.5, 1.0)
    axes[1].barh(order["run"], order["val_loss"], color="tab:orange")
    axes[1].set_xlabel("best validation loss (lower = better)")
    fig.suptitle("Experiment comparison (selection on validation only)")
    fig.tight_layout()
    out = FIGURES_DIR / "experiment_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    best = runs.iloc[0]
    print(f"\nBest by validation AUC: {best['run']}  (val_auc={best['val_auc']:.3f}, "
          f"val_loss={best['val_loss']:.3f}, test_auc={best['test_auc']:.3f})")


if __name__ == "__main__":
    main()
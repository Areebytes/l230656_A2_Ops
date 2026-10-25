"""
VisionGuard - Task 2.5: Evaluation.

1. Predict on VALIDATION -> choose the decision threshold there.
2. Predict on TEST once  -> report final metrics at that threshold.
Outputs metric table, confusion matrices, ROC curve, threshold plots,
and per-image test predictions (used by 2.6 error analysis).
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,  # noqa: E402
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR  # noqa: E402
from data_pipeline import make_dataset  # noqa: E402

MODELS = {
    "cnn_baseline": "Baseline CNN",
    "transfer_mobilenetv2": "MobileNetV2 (fine-tuned)",
}
CLASS_NAMES = ["Real", "AI"]  # label 0, label 1


def predict(model, split):
    """Probabilities in the same order as the split CSV
    (no shuffling for val/test, and tf.data map keeps order)."""
    df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    df["prob_ai"] = model.predict(make_dataset(split), verbose=0).ravel()
    return df


def choose_threshold(y, p, strategy, max_fpr):
    """Pick the threshold on VALIDATION data only."""
    if strategy == "max_fpr":
        # Highest AI recall (TPR) while real->AI errors (FPR) stay <= max_fpr
        fpr, tpr, thr = roc_curve(y, p)
        ok = np.where(fpr <= max_fpr)[0]
        best = ok[np.argmax(tpr[ok])]
        return float(min(thr[best], 1.0))  # roc_curve's first threshold is +inf
    if strategy == "youden":
        # Best balance of TPR and (1 - FPR)
        fpr, tpr, thr = roc_curve(y, p)
        return float(min(thr[np.argmax(tpr - fpr)], 1.0))
    if strategy == "f1":
        grid = np.linspace(0.05, 0.95, 91)
        scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(scores))])
    if strategy == "default":
        return 0.5
    raise ValueError(strategy)


def compute_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(t, 4),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "fpr_real_called_ai": fp / (fp + tn),
        "fnr_ai_called_real": fn / (fn + tp),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def plot_threshold_choice(y, p, t, title, out):
    """Validation precision / recall / FPR across thresholds (justifies the choice)."""
    grid = np.linspace(0.01, 0.99, 99)
    prec, rec, fpr = [], [], []
    for g in grid:
        m = compute_metrics(y, p, g)
        prec.append(m["precision"]); rec.append(m["recall"]); fpr.append(m["fpr_real_called_ai"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, prec, label="precision (AI)")
    ax.plot(grid, rec, label="recall (AI)")
    ax.plot(grid, fpr, label="FPR: real called AI")
    ax.axvline(t, color="red", linestyle="--", label=f"chosen threshold = {t:.3f}")
    ax.set_xlabel("threshold on P(AI)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y, p, t, title, out):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, (p >= t).astype(int), labels=[0, 1],
                                            display_labels=CLASS_NAMES, ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["max_fpr", "youden", "f1", "default"],
                        default="max_fpr")
    parser.add_argument("--max-fpr", type=float, default=0.10,
                        help="max share of real images called AI (max_fpr strategy)")
    args = parser.parse_args()

    rows, thresholds = [], {}
    roc_fig, roc_ax = plt.subplots(figsize=(6, 6))

    for run_name, nice_name in MODELS.items():
        path = MODELS_DIR / run_name / "best.keras"
        print(f"\n=== {nice_name} ({path}) ===")
        model = keras.models.load_model(path, compile=False)  # inference only

        # 1) threshold from VALIDATION
        val = predict(model, "val")
        t = choose_threshold(val["label"].to_numpy(), val["prob_ai"].to_numpy(),
                             args.strategy, args.max_fpr)
        thresholds[run_name] = t
        val_m = compute_metrics(val["label"].to_numpy(), val["prob_ai"].to_numpy(), t)
        print(f"Threshold ({args.strategy}) chosen on validation: {t:.4f}")
        plot_threshold_choice(val["label"].to_numpy(), val["prob_ai"].to_numpy(), t,
                              f"{nice_name}: validation threshold choice",
                              FIGURES_DIR / f"{run_name}_threshold_val.png")

        # 2) TEST, once
        test = predict(model, "test")
        y, p = test["label"].to_numpy(), test["prob_ai"].to_numpy()
        test["pred"] = (p >= t).astype(int)
        test["correct"] = test["pred"] == test["label"]
        test.to_csv(REPORTS_DIR / f"{run_name}_test_predictions.csv", index=False)

        test_m = compute_metrics(y, p, t)
        rows.append({"model": nice_name, "split": "val", **val_m})
        rows.append({"model": nice_name, "split": "test", **test_m})

        plot_confusion(y, p, t, f"{nice_name} - test (t={t:.3f})",
                       FIGURES_DIR / f"{run_name}_confusion_test.png")
        fpr, tpr, _ = roc_curve(y, p)
        line, = roc_ax.plot(fpr, tpr, label=f"{nice_name} (AUC={test_m['roc_auc']:.3f})")
        roc_ax.scatter(test_m["fpr_real_called_ai"], test_m["recall"], color=line.get_color(),
                       zorder=5, label=f"operating point t={t:.3f}")

    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random guess")
    roc_ax.set_xlabel("False positive rate (real called AI)")
    roc_ax.set_ylabel("True positive rate (AI caught)")
    roc_ax.set_title("ROC curve - test set")
    roc_ax.legend(fontsize=8, loc="lower right")
    roc_fig.tight_layout()
    roc_fig.savefig(FIGURES_DIR / "roc_test.png", dpi=150, bbox_inches="tight")
    plt.close(roc_fig)

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "evaluation_metrics.csv", index=False)
    (REPORTS_DIR / "thresholds.json").write_text(json.dumps(
        {"strategy": args.strategy, "max_fpr": args.max_fpr, "thresholds": thresholds}, indent=2))

    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n" + table.to_string(index=False))
    print(f"\nSaved metrics, thresholds.json, figures and test predictions to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
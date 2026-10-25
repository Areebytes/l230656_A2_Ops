"""
VisionGuard - Task 2.6: Error analysis.

Picks the most confidently misclassified TEST images (balanced FP / FN),
shows them as the user uploaded them AND as the model saw them, and breaks
error rates down by format / mode / size to test hypotheses.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from config import DATA_DIR, FIGURES_DIR, REPORTS_DIR  # noqa: E402
from data_pipeline import decode_image, resize_image  # noqa: E402


def load_predictions(run_name):
    preds = pd.read_csv(REPORTS_DIR / f"{run_name}_test_predictions.csv")
    audit = pd.read_csv(REPORTS_DIR / "audit.csv")[
        ["path", "width", "height", "format", "mode", "file_size"]]
    df = preds.merge(audit, on="path", how="left")
    df["error_type"] = np.where(df["label"] == 1, "FN: AI called real", "FP: real called AI")
    # how sure the model was of its (wrong) answer
    df["wrong_confidence"] = np.where(df["pred"] == 1, df["prob_ai"], 1 - df["prob_ai"])
    df["short_side"] = df[["width", "height"]].min(axis=1)
    return df


def pick_errors(errors, n):
    """Most confident mistakes, half FP and half FN where possible."""
    by_conf = lambda d: d.sort_values("wrong_confidence", ascending=False)  # noqa: E731
    fp, fn = by_conf(errors[errors["label"] == 0]), by_conf(errors[errors["label"] == 1])
    chosen = pd.concat([fp.head(n // 2), fn.head(n - n // 2)])
    if len(chosen) < n:  # one type has too few -> fill with the other
        chosen = pd.concat([chosen, by_conf(errors.drop(chosen.index)).head(n - len(chosen))])
    return by_conf(chosen).reset_index(drop=True)


def title_for(i, row):
    kind = "FP" if row["label"] == 0 else "FN"
    return (f"#{i + 1} {kind}  P(AI)={row['prob_ai']:.2f}\n"
            f"{int(row['width'])}x{int(row['height'])} {row['format']} {row['mode']}")


def grid(chosen, loader, suptitle, out):
    cols = 5
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if i < len(chosen):
            row = chosen.iloc[i]
            ax.imshow(loader(DATA_DIR / row["path"]))
            ax.set_title(title_for(i, row), fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def load_original(path):
    """Roughly what the user uploaded (shrunk for display, aspect ratio kept)."""
    with Image.open(path) as img:
        img.draft("RGB", (400, 400))
        img = img.convert("RGB")
        img.thumbnail((400, 400))
        return np.asarray(img)


def load_model_view(path):
    """Exactly what the model received: same decode + resize as training."""
    return resize_image(decode_image(tf.io.read_file(str(path)))).numpy()


def error_rate_table(df, column):
    t = df.groupby(column, observed=True).agg(images=("correct", "size"),
                                              errors=("correct", lambda c: int((~c).sum())))
    t["error_rate"] = (t["errors"] / t["images"]).round(3)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="transfer_mobilenetv2")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    df = load_predictions(args.run_name)
    errors = df[~df["correct"]]
    print(f"Test images: {len(df)}  misclassified: {len(errors)} "
          f"({(errors['label'] == 0).sum()} FP, {(errors['label'] == 1).sum()} FN)")

    chosen = pick_errors(errors, args.n)
    grid(chosen, load_original, f"{len(chosen)} most confident test errors - original",
         FIGURES_DIR / f"{args.run_name}_errors_original.png")
    grid(chosen, load_model_view, f"Same images as the model saw them (input pipeline)",
         FIGURES_DIR / f"{args.run_name}_errors_model_view.png")

    # Table to fill in: one hypothesis per image
    out_cols = ["path", "error_type", "prob_ai", "wrong_confidence",
                "width", "height", "format", "mode", "file_size"]
    table = chosen[out_cols].copy()
    table.insert(0, "id", range(1, len(table) + 1))
    table["hypothesis"] = ""
    table.to_csv(REPORTS_DIR / f"{args.run_name}_error_analysis.csv", index=False)
    print("\n" + table.drop(columns=["hypothesis"]).to_string(index=False))

    # Evidence for/against hypotheses across the WHOLE test set
    df["size_bucket"] = pd.cut(df["short_side"], [0, 300, 600, 1200, 1e9],
                               labels=["<300px", "300-600px", "600-1200px", ">1200px"])
    for col in ["class", "format", "mode", "size_bucket"]:
        print(f"\n=== Error rate by {col} ===")
        print(error_rate_table(df, col).to_string())


if __name__ == "__main__":
    main()
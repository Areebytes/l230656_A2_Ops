"""
VisionGuard - Task 1.3: Visual EDA.

Reads the audit table (no re-scanning of metadata) and produces:
  - a grid of random sample images per class
  - per-class histograms of width, height, file size and aspect ratio
All figures are saved to reports/figures/.
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # draw straight to files; no pop-up windows needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from config import DATA_DIR, FIGURES_DIR, REPORTS_DIR  # noqa: E402

SEED = 42  # same samples every run -> reproducible figures
THUMB = (256, 256)


def load_clean_table() -> pd.DataFrame:
    """Audit metadata restricted to the cleaned files we will actually train on."""
    audit_path = REPORTS_DIR / "audit.csv"
    clean_path = REPORTS_DIR / "clean_files.csv"
    if not audit_path.exists() or not clean_path.exists():
        raise SystemExit("Run src/audit.py first: audit.csv / clean_files.csv missing.")

    audit = pd.read_csv(audit_path)
    clean_paths = set(pd.read_csv(clean_path)["path"])
    df = audit[audit["path"].isin(clean_paths)].copy()

    df["file_size_kb"] = df["file_size"] / 1024
    df["aspect_ratio"] = df["width"] / df["height"]  # >1 landscape, <1 portrait
    return df


def sample_grid(df: pd.DataFrame, n_per_class: int) -> None:
    """One row per class, n random images each, titled with size and format."""
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(2.3 * n_per_class, 2.7 * len(classes)))
    axes = np.atleast_2d(axes)

    for r, cls in enumerate(classes):
        sample = df[df["class"] == cls].sample(n=n_per_class, random_state=SEED)
        for c, (_, row) in enumerate(sample.iterrows()):
            ax = axes[r, c]
            with Image.open(DATA_DIR / row["path"]) as img:
                img.draft("RGB", THUMB)       # JPEG shortcut: decode at reduced size (fast)
                img = img.convert("RGB")      # L / P / RGBA -> 3 channels for display
                img.thumbnail(THUMB)          # shrink, keeping aspect ratio
                ax.imshow(img)
            ax.set_title(f"{int(row['width'])}x{int(row['height'])} {row['format']}",
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(cls, fontsize=12, fontweight="bold")

    fig.suptitle(f"Random samples per class (seed={SEED})", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "sample_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def class_histogram(df: pd.DataFrame, column: str, xlabel: str,
                    filename: str, log_scale: bool) -> None:
    """Overlaid per-class histogram, normalized so classes of different size compare fairly."""
    values = df[column]
    if log_scale:
        # Data spans ~100 px to ~8500 px: log-spaced bins stop a few giant
        # images from squashing everything else into one bar.
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
    else:
        bins = np.linspace(values.min(), values.max(), 40)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls in sorted(df["class"].unique()):
        vals = df.loc[df["class"] == cls, column]
        # weights -> each bar shows the FRACTION of that class, not a raw count,
        # because the classes have different sizes (538 vs 436)
        line = ax.hist(vals, bins=bins, alpha=0.5,
                       weights=np.ones(len(vals)) / len(vals),
                       label=f"{cls} (n={len(vals)}, median={vals.median():.0f})")
        ax.axvline(vals.median(), linestyle="--", linewidth=1.5,
                   color=line[2][0].get_facecolor())  # median line in matching colour

    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of class")
    ax.set_title(f"{xlabel} by class")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Visual EDA for VisionGuard.")
    parser.add_argument("--n-per-class", type=int, default=6)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_table()
    print(f"Loaded {len(df)} clean images: {df['class'].value_counts().to_dict()}")

    sample_grid(df, args.n_per_class)
    class_histogram(df, "width", "Width (px)", "hist_width.png", log_scale=True)
    class_histogram(df, "height", "Height (px)", "hist_height.png", log_scale=True)
    class_histogram(df, "file_size_kb", "File size (KB)", "hist_file_size.png", log_scale=True)
    # Extra figure (not required): directly informs the resize decision in Task 2.1
    class_histogram(df, "aspect_ratio", "Aspect ratio (width / height)",
                    "hist_aspect_ratio.png", log_scale=False)


if __name__ == "__main__":
    main()
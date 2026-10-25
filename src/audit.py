"""
VisionGuard - Task 1.2: Dataset audit.

Walks the dataset, records per-image metadata, detects unreadable files
and exact duplicates, prints summary tables, and writes:
  reports/audit.csv        -> one row per file (the full evidence table)
  reports/clean_files.csv  -> usable, de-duplicated files for building splits
"""
import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from config import DATA_DIR, REPORTS_DIR

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF info logs after start-up
import tensorflow as tf  # noqa: E402

CHUNK_SIZE = 64 * 1024  # read files in 64 KB pieces


def sha256_of(path: Path) -> str:
    """Hash the raw bytes in chunks so large files never load fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_with_pil(path: Path) -> dict:
    """Return size/mode/format, or record why the file could not be read.

    Image.open() is lazy (reads only the header), so a truncated file can
    still 'open'. verify() checks integrity but leaves the object unusable,
    so we reopen and load() to force a full decode and read attributes.
    """
    info = {"width": None, "height": None, "mode": None,
            "format": None, "pil_ok": False, "pil_error": ""}
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            info.update(width=img.width, height=img.height,
                        mode=img.mode, format=img.format, pil_ok=True)
    except Exception as e:  # any failure = unreadable
        info["pil_error"] = f"{type(e).__name__}: {e}"
    return info


def decodes_with_tf(path: Path) -> tuple[bool, str]:
    """Check the file decodes with TensorFlow, which the tf.data pipeline will use.
    PIL supports more formats than TF, so PIL-readable != pipeline-safe."""
    try:
        raw = tf.io.read_file(str(path))
        tf.io.decode_image(raw, channels=3, expand_animations=False)
        return True, ""
    except Exception as e:
        return False, type(e).__name__


def build_table(root: Path) -> pd.DataFrame:
    """One row per file. The label is the class folder directly under root."""
    if not root.is_dir():
        raise SystemExit(f"Data folder not found: {root}")
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {root}")

    rows = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            row = {
                # POSIX-style relative paths work on Windows, Linux and in Docker
                "path": path.relative_to(root).as_posix(),
                "class": class_dir.name,
                "extension": path.suffix.lower(),
                "file_size": path.stat().st_size,
                "sha256": sha256_of(path),
            }
            row.update(inspect_with_pil(path))
            row["tf_ok"], row["tf_error"] = decodes_with_tf(path)
            rows.append(row)
            if len(rows) % 200 == 0:
                print(f"  audited {len(rows)} files...")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, out: Path) -> None:
    print("\n=== Files per class ===")
    print(df["class"].value_counts().to_string())

    readable = df[df["pil_ok"]]
    print("\n=== Width / height / file size per class (readable files) ===")
    stats = readable.groupby("class")[["width", "height", "file_size"]].agg(
        ["min", "median", "max"])
    print(stats.to_string())

    print("\n=== Format per class ===")
    print(pd.crosstab(readable["class"], readable["format"]).to_string())
    print("\n=== Colour mode per class ===")
    print(pd.crosstab(readable["class"], readable["mode"]).to_string())

    # --- unreadable files ---
    pil_bad = df[~df["pil_ok"]]
    tf_bad = df[df["pil_ok"] & ~df["tf_ok"]]
    print(f"\n=== Unreadable ===\nPIL could not decode: {len(pil_bad)}"
          f"\nPIL OK but TensorFlow could not decode: {len(tf_bad)}")
    if len(pil_bad) or len(tf_bad):
        print(pd.concat([pil_bad, tf_bad])[
            ["path", "extension", "pil_error", "tf_error"]].to_string(index=False))

    # --- exact duplicates (same SHA-256 = byte-identical) ---
    dup = df[df.duplicated("sha256", keep=False)]
    classes_per_hash = dup.groupby("sha256")["class"].nunique()
    cross = classes_per_hash[classes_per_hash > 1].index
    within = classes_per_hash[classes_per_hash == 1].index
    extra_copies = len(dup) - dup["sha256"].nunique()
    print("\n=== Duplicates ===")
    print(f"Duplicate groups: {dup['sha256'].nunique()} "
          f"({len(within)} within one class, {len(cross)} across classes)")
    print(f"Redundant extra copies: {extra_copies}")
    if len(cross):
        print("Cross-class duplicates (label noise!):")
        print(dup[dup["sha256"].isin(cross)][["class", "path"]].to_string(index=False))

    # --- clean list: readable by both, no label conflicts, one copy per hash ---
    clean = df[df["pil_ok"] & df["tf_ok"] & ~df["sha256"].isin(cross)]
    clean = clean.drop_duplicates("sha256", keep="first")
    clean[["path", "class", "sha256"]].to_csv(out / "clean_files.csv", index=False)
    print("\n=== Clean set (kept / total) ===")
    print(pd.concat([clean["class"].value_counts().rename("kept"),
                     df["class"].value_counts().rename("total")], axis=1).to_string())


def main():
    parser = argparse.ArgumentParser(description="Audit the VisionGuard dataset.")
    parser.add_argument("--root", default=str(DATA_DIR),
                        help="folder that directly contains one subfolder per class")
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Auditing {root.resolve()} ...")
    df = build_table(root)
    df.to_csv(out / "audit.csv", index=False)
    summarize(df, out)
    print(f"\nWrote {out / 'audit.csv'} and {out / 'clean_files.csv'}")


if __name__ == "__main__":
    main()
"""
VisionGuard - split lists.

Stratified 70/15/15 split of the cleaned, de-duplicated file list.
Writes splits/train.csv, splits/val.csv, splits/test.csv.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASS_TO_LABEL, REPORTS_DIR, SEED, SPLITS_DIR

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main():
    clean = pd.read_csv(REPORTS_DIR / "clean_files.csv")

    unknown = set(clean["class"]) - set(CLASS_TO_LABEL)
    if unknown:
        raise SystemExit(f"Classes not in CLASS_TO_LABEL: {unknown}")
    clean["label"] = clean["class"].map(CLASS_TO_LABEL)

    # 1st cut: carve out the test set. stratify keeps the AI/real ratio identical.
    train_val, test = train_test_split(
        clean, test_size=TEST_FRAC, stratify=clean["label"], random_state=SEED)

    # 2nd cut: val is 15% of the WHOLE dataset, i.e. 0.15 / 0.85 of what remains.
    train, val = train_test_split(
        train_val, test_size=VAL_FRAC / (1 - TEST_FRAC),
        stratify=train_val["label"], random_state=SEED)

    # Leakage check: no image (by content hash) may appear in two splits.
    sets = {"train": set(train["sha256"]), "val": set(val["sha256"]), "test": set(test["sha256"])}
    assert not (sets["train"] & sets["val"]), "train/val overlap!"
    assert not (sets["train"] & sets["test"]), "train/test overlap!"
    assert not (sets["val"] & sets["test"]), "val/test overlap!"

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.sort_values("path")  # stable order -> clean git diffs
        part[["path", "class", "label", "sha256"]].to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        counts = part["class"].value_counts()
        rows.append({"split": name, "total": len(part), **counts.to_dict(),
                     "AI share": f"{part['label'].mean():.1%}"})

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote split lists to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
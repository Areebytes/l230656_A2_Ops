"""
VisionGuard - Task 2.7 apply: tune fine-tuning LR, dropout and unfrozen depth.

Stage 1 (frozen backbone) is trained ONCE and shared; every trial loads those
weights and runs only stage 2 with its own hyperparameters.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import keras_tuner as kt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import MODELS_DIR, REPORTS_DIR, SEED  # noqa: E402
from data_pipeline import make_dataset  # noqa: E402
from models import build_transfer_model, layers_from, unfreeze_top  # noqa: E402
from train import compile_model, set_seeds  # noqa: E402

TUNER_DIR = MODELS_DIR / "tuner"
STAGE1_WEIGHTS = TUNER_DIR / "stage1.weights.h5"
HPS = ["ft_lr", "dropout", "unfreeze_blocks"]


def ensure_stage1_weights(train_ds, val_ds, epochs):
    """Train the frozen-backbone stage once; reuse it for every trial."""
    if STAGE1_WEIGHTS.exists():
        print(f"Reusing stage-1 weights: {STAGE1_WEIGHTS}")
        return
    TUNER_DIR.mkdir(parents=True, exist_ok=True)
    print("Training shared stage 1 (backbone frozen)...")
    model, _ = build_transfer_model(dropout=0.3)
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                       restore_best_weights=True)])
    model.save_weights(STAGE1_WEIGHTS)
    print(f"Saved {STAGE1_WEIGHTS}")


class VisionGuardHyperModel(kt.HyperModel):
    def build(self, hp):
        ft_lr = hp.Float("ft_lr", min_value=1e-5, max_value=3e-4, sampling="log")
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        blocks = hp.Choice("unfreeze_blocks", [1, 2, 3, 4])

        model, base = build_transfer_model(dropout=dropout)
        model.load_weights(STAGE1_WEIGHTS)       # dropout has no weights, so this always fits
        # last k blocks of MobileNetV2 = from block_(17-k)_expand to the end
        unfreeze_top(base, layers_from(base, f"block_{17 - blocks}_expand"))
        compile_model(model, ft_lr)
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10, help="stage-2 epochs per trial")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", help="discard previous trials")
    args = parser.parse_args()

    set_seeds()
    train_ds = make_dataset("train", training=True)
    val_ds = make_dataset("val")
    ensure_stage1_weights(train_ds, val_ds, args.stage1_epochs)

    tuner = kt.RandomSearch(VisionGuardHyperModel(), objective="val_loss",
                            max_trials=args.trials, seed=SEED,
                            directory=str(TUNER_DIR), project_name="visionguard",
                            overwrite=args.overwrite)
    tuner.search_space_summary()
    tuner.search(train_ds, validation_data=val_ds, epochs=args.epochs,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])

    # ---- results table ----
    trials = tuner.oracle.get_best_trials(num_trials=args.trials)
    table = pd.DataFrame([{"trial": t.trial_id, **{h: t.hyperparameters.get(h) for h in HPS},
                           "best_val_loss": t.score} for t in trials])
    table["n_unfrozen_layers"] = None
    _, base = build_transfer_model()
    table["n_unfrozen_layers"] = [layers_from(base, f"block_{17 - b}_expand")
                                  for b in table["unfreeze_blocks"]]
    table.to_csv(REPORTS_DIR / "tuner_visionguard_results.csv", index=False)
    pd.set_option("display.float_format", "{:.5f}".format)
    print("\n=== All trials (best first) ===")
    print(table.to_string(index=False))

    # ---- which hyperparameter mattered most? ----
    print("\n=== Rank correlation with val_loss (|value| near 1 = strong effect) ===")
    for h in HPS:
        x = np.log10(table[h]) if h == "ft_lr" else table[h]
        print(f"  {h:16s} {x.corr(table['best_val_loss'], method='spearman'):+.2f}")
    print("\n=== Mean val_loss per unfreeze depth ===")
    print(table.groupby("unfreeze_blocks")["best_val_loss"].agg(["mean", "min", "count"]))

    best = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBest: ft_lr={best.get('ft_lr'):.2e}  dropout={best.get('dropout'):.1f}  "
          f"unfreeze_blocks={best.get('unfreeze_blocks')}")


if __name__ == "__main__":
    main()
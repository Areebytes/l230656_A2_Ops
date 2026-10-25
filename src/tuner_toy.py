"""
VisionGuard - Task 2.7 toy: Keras Tuner (Hyperband) on Fashion-MNIST.
Searches learning rate and number of hidden units of a tiny dense network.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import keras_tuner as kt  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from config import MODELS_DIR, REPORTS_DIR, SEED  # noqa: E402

TUNER_DIR = MODELS_DIR / "tuner"


def build_model(hp):
    """Called once per trial with a different set of hyperparameter values."""
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(units, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),  # 10 clothing classes
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    tuner = kt.Hyperband(build_model, objective="val_accuracy", max_epochs=10, factor=3,
                         seed=SEED, directory=str(TUNER_DIR),
                         project_name="fashion_mnist_toy", overwrite=True)
    tuner.search_space_summary()
    tuner.search(x_train, y_train, validation_split=0.2,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])

    # Results table: one row per trial
    trials = tuner.oracle.get_best_trials(num_trials=1000)
    rows = [{"trial": t.trial_id,
             "units": t.hyperparameters.get("units"),
             "learning_rate": t.hyperparameters.get("learning_rate"),
             "epochs": t.hyperparameters.values.get("tuner/epochs"),
             "val_accuracy": t.score} for t in trials]
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "tuner_toy_results.csv", index=False)
    print("\n=== Top 10 trials ===")
    print(table.head(10).to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBest: units={best_hp.get('units')}  learning_rate={best_hp.get('learning_rate'):.5f}")
    best_model = tuner.get_best_models(1)[0]
    _, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Best model test accuracy: {test_acc:.4f}  ({len(table)} trials run)")


if __name__ == "__main__":
    main()
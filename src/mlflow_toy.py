"""
VisionGuard - Task 3.1: MLflow tracking basics on a tiny Keras model (Fashion-MNIST).

Logs parameters, per-epoch metrics, tags, a plot and the model.
Run twice with different learning rates, then compare in the MLflow UI.

Where runs go is decided ONLY by the tracking URI:
  - MLFLOW_TRACKING_URI set (e.g. DagsHub in 3.2) -> that server
  - not set                                       -> local sqlite file mlflow.db
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import keras  # noqa: E402
import mlflow  # noqa: E402
from keras import layers  # noqa: E402

from config import PROJECT_ROOT, SEED  # noqa: E402

EXPERIMENT = "l230656_A2_Ops_mlflow_toy"


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual logging: after every epoch, send each metric with step = epoch number."""

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            mlflow.log_metric(name, float(value), step=epoch + 1)


def build_model(units, lr):
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    print("Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment(EXPERIMENT)          # created on first use
    keras.utils.set_random_seed(SEED)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train[:20000] / 255.0, x_test / 255.0   # subset: fast on CPU
    y_train = y_train[:20000]

    with mlflow.start_run(run_name=f"lr_{args.lr:g}") as run:
        # Parameters: the inputs that define this run
        mlflow.log_params({"learning_rate": args.lr, "units": args.units,
                           "epochs": args.epochs, "batch_size": args.batch_size,
                           "train_samples": len(x_train), "seed": SEED})
        # Tags: descriptive metadata
        mlflow.set_tags({"stage": "toy", "dataset": "fashion_mnist", "framework": "keras"})

        model = build_model(args.units, args.lr)
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=args.epochs,
                            batch_size=args.batch_size, callbacks=[MLflowEpochLogger()],
                            verbose=2)

        # Final metrics (single values, no step)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        # Artifact: a plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history["loss"], label="train loss")
        ax.plot(history.history["val_loss"], label="val loss")
        ax.set_xlabel("epoch"); ax.set_title(f"lr={args.lr:g}"); ax.legend()
        mlflow.log_figure(fig, "plots/loss_curve.png")
        plt.close(fig)

        # The model itself (MLflow's TensorFlow/Keras flavor)
        mlflow.tensorflow.log_model(model, name="model")

        print(f"\nRun ID: {run.info.run_id}   test_accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
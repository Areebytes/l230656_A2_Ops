"""
VisionGuard - shared MLflow helpers (Task 3.3).
"""
import importlib.metadata
import os
import subprocess

import keras
import mlflow
import yaml

from config import PROJECT_ROOT, SEED

EXPERIMENT_NAME = "l230656_A2_Ops"
REGISTERED_MODEL = "l230656_A2_Ops"


def setup_mlflow(experiment=EXPERIMENT_NAME):
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set - check your .env")
    mlflow.set_experiment(experiment)
    print("Tracking URI:", mlflow.get_tracking_uri())


def git_commit():
    """Current commit + whether tracked files have uncommitted changes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=PROJECT_ROOT, text=True).strip()
        return commit, bool(dirty)
    except Exception:
        return "unknown", True


def dvc_data_version():
    """The md5 in data/raw.dvc identifies the exact dataset version."""
    meta = yaml.safe_load((PROJECT_ROOT / "data" / "raw.dvc").read_text())
    out = meta["outs"][0]
    return out.get("md5", "unknown"), out.get("nfiles", "unknown")


def common_tags():
    commit, dirty = git_commit()
    md5, nfiles = dvc_data_version()
    if dirty:
        print("WARNING: uncommitted changes - this run's code is not exactly the tagged commit")
    return {"git_commit": commit, "git_dirty": str(dirty),
            "dvc_data_md5": md5, "dvc_data_nfiles": str(nfiles),
            "seed": str(SEED), "project": EXPERIMENT_NAME}


class MLflowEpochLogger(keras.callbacks.Callback):
    """MANUAL logging: all Keras metrics after each epoch, one request per epoch.
    The epoch index continues across stages (initial_epoch), so curves are continuous."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def log_environment():
    """Package list: the pinned file AND what is actually installed right now."""
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}"
                  for d in importlib.metadata.distributions() if d.metadata["Name"])
    mlflow.log_text("\n".join(pkgs), "environment/installed_packages.txt")
    mlflow.log_artifact(str(PROJECT_ROOT / "requirements.txt"), "environment")
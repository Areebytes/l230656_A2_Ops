"""
Central project configuration.

Every path is anchored to the project root (so scripts work no matter which
folder you run them from) and can be overridden with an environment variable
(so Docker, the cloud VM or CI can point to different locations without
code changes).
"""
import os
from pathlib import Path

# src/config.py -> parent is src/, parent.parent is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(var_name: str, default: Path) -> Path:
    """Use the environment variable if set, otherwise the default path."""
    value = os.getenv(var_name)
    return Path(value).resolve() if value else default


DATA_DIR    = _path_from_env("DATA_DIR",    PROJECT_ROOT / "data" / "raw")
REPORTS_DIR = _path_from_env("REPORTS_DIR", PROJECT_ROOT / "reports")
SPLITS_DIR  = _path_from_env("SPLITS_DIR",  PROJECT_ROOT / "splits")
FIGURES_DIR = _path_from_env("FIGURES_DIR", REPORTS_DIR / "figures")

# ---------------- Training settings (overridable per experiment) ----------------
SEED        = int(os.getenv("SEED", 42))
IMG_SIZE    = int(os.getenv("IMG_SIZE", 160))     # square input, pixels
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 32))
RESIZE_MODE = os.getenv("RESIZE_MODE", "crop")    # "crop" | "pad" | "stretch"

# Positive class = AI. The model's sigmoid output is P(image is AI-generated).
CLASS_TO_LABEL = {"RealArt": 0, "AiArtData": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

MODELS_DIR = _path_from_env("MODELS_DIR", PROJECT_ROOT / "models")
"""Data and output root for the produce experiments.

RIPENESS_TRAINING_DIR sets the root (default: <repo>/models/training). Sub-folders keep their names: produce/ lives under it.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(
    os.environ.get("RIPENESS_TRAINING_DIR") or REPO_ROOT / "models" / "training"
)
ROOT = str(TRAINING_DIR / "produce")

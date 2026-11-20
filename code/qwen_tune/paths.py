"""Paths for training and scoring: the single source for every location.

The RIPEWISE_ prefix is the project's name; the same name appears in the trained prompts.

Environment variables (see README.md):
  RIPEWISE_TRAINING_DIR    data and output root, default <repo>/models/training
  BANANA_SPLITS            banana split file, default <training>/splits.csv
  RIPEWISE_BANANA_DATASET  the BananaImageBD folder, default the folder of that name at the repo root
  RIPEWISE_QWEN_BASE       Qwen3.5-0.8B snapshot folder, default the pinned snapshot under HF_HOME
  RIPEWISE_LLAMA_CPP       llama.cpp checkout (convert_lora_to_gguf.py), default <training>/qwen-tune/llama.cpp
  HF_HOME                  Hugging Face cache, default <training>/hf-cache
"""

import os
from pathlib import Path

BANANA_NAME = "BananaImageBD A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh"
QWEN_SNAPSHOT = "2fc06364715b967f1860aea9cf38778875588b17"

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(
    os.environ.get("RIPEWISE_TRAINING_DIR") or REPO_ROOT / "models" / "training"
)
QWEN_DIR = TRAINING_DIR / "qwen-tune"
CROPS = TRAINING_DIR / "produce" / "crops"
SPLITS_CSV = Path(os.environ.get("BANANA_SPLITS") or TRAINING_DIR / "splits.csv")

HF_HOME = Path(os.environ.setdefault("HF_HOME", str(TRAINING_DIR / "hf-cache")))
BASE_MODEL = Path(
    os.environ.get("RIPEWISE_QWEN_BASE")
    or HF_HOME / "hub" / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / QWEN_SNAPSHOT
)

LLAMA_CPP_DIR = Path(os.environ.get("RIPEWISE_LLAMA_CPP") or QWEN_DIR / "llama.cpp")

_banana = os.environ.get("RIPEWISE_BANANA_DATASET") or BANANA_NAME
BANANA_DATASET = Path(_banana) if Path(_banana).is_absolute() else REPO_ROOT / _banana
_MARKER = "/" + BANANA_NAME + "/"


def banana_path(p):
    """Map a photo path stored in splits.csv to the same file under BANANA_DATASET."""
    i = p.find(_MARKER)
    if i < 0 and Path(p).is_file():  # photos from the added banana datasets
        return p
    if i < 0:
        raise ValueError(f"photo is not under the BananaImageBD folder: {p}")
    return str(BANANA_DATASET / p[i + len(_MARKER) :])

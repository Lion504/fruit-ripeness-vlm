"""Shared data loading and metrics for the ripeness classifier experiments.

Environment variables (see README.md):
  RIPENESS_TRAINING_DIR  data and output root, default <repo>/models/training
  RIPENESS_BANANA_DATASET  the BananaImageBD folder, default the folder of that name at the repo root
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

BANANA_NAME = "BananaImageBD A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh"
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(
    os.environ.get("RIPENESS_TRAINING_DIR") or REPO_ROOT / "models" / "training"
)
_banana = os.environ.get("RIPENESS_BANANA_DATASET") or BANANA_NAME
BANANA_DATASET = Path(_banana) if Path(_banana).is_absolute() else REPO_ROOT / _banana

SPLITS_CSV = str(TRAINING_DIR / "splits.csv")
OUT_DIR = str(TRAINING_DIR / "classifier")
CLASSES = ["unripe", "turning", "ripe", "overripe"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_MARKER = "/" + BANANA_NAME + "/"


def photo_path(p):
    """Map a photo path stored in splits.csv to the same file under BANANA_DATASET."""
    i = p.find(_MARKER)
    if i < 0:
        raise ValueError(f"photo is not under the BananaImageBD folder: {p}")
    return str(BANANA_DATASET / p[i + len(_MARKER) :])


def read_splits():
    """splits.csv with the path column pointing at BANANA_DATASET."""
    df = pd.read_csv(SPLITS_CSV)
    df["path"] = [photo_path(p) for p in df["path"]]
    return df


def load_split(split):
    """Return (paths, labels as ints) for split in train/val/test or 'trainval'."""
    df = read_splits()
    if split == "trainval":
        df = df[df.split.isin(["train", "val"])]
    else:
        df = df[df.split == split]
    y = df.label.map(CLASSES.index).to_numpy()
    return df.path.tolist(), y, df.reset_index(drop=True)


def load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


def report(y, pred, name=""):
    """Print accuracy, per-class correct, macro F1, confusion matrix; return dict."""
    y, pred = np.asarray(y), np.asarray(pred)
    cm = confusion_matrix(y, pred, labels=range(4))
    acc = float((y == pred).mean())
    f1 = float(f1_score(y, pred, average="macro", labels=range(4)))
    per = {c: int(cm[i, i]) for i, c in enumerate(CLASSES)}
    totals = cm.sum(1)
    print(
        f"[{name}] acc {int((y == pred).sum())}/{len(y)} = {acc:.4f}  macroF1 {f1:.4f}"
    )
    print(
        "  per-class correct: "
        + ", ".join(f"{c} {per[c]}/{totals[i]}" for i, c in enumerate(CLASSES))
    )
    print("  confusion (rows=true, cols=pred; order " + "/".join(CLASSES) + ")")
    for i, row in enumerate(cm):
        print("   ", CLASSES[i].ljust(9), row.tolist())
    return dict(acc=acc, f1=f1, per=per, cm=cm.tolist())

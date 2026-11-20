"""Build a small-in-frame banana VALIDATION list from the banana val split (no test photos).

usage: build_banana_val_small.py OUT_DIR [SEED]

Each val photo is shrunk so its long side is 20-35% of a 256 px square and placed at a random spot on a
generated texture (wavy stripes or noise in a random colour), the same style as the small-in-frame test photos.
Writes OUT_DIR/<i>.png and OUT_DIR/val_small.jsonl (path, label, source=val_small_<source>).
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_tune"))
from paths import SPLITS_CSV, banana_path  # noqa: E402


def texture(rng, size=256):
    col = np.array([rng.randrange(40, 230) for _ in range(3)], np.float32)
    a = np.ones((size, size, 3), np.float32) * col
    if rng.random() < 0.6:
        period = rng.uniform(8, 30)
        phase = np.sin(np.arange(size) / rng.uniform(10, 40)) * rng.uniform(2, 8)
        rows = np.arange(size)[:, None] + phase[None, :]
        a *= (1 + 0.25 * np.sin(rows / period * 6.28))[:, :, None]
    a += np.random.default_rng(rng.randrange(10**9)).normal(0, rng.uniform(5, 20), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main(out_dir, seed=35):
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    rows = [r for r in csv.DictReader(open(SPLITS_CSV)) if r["split"] == "val"]
    with open(out / "val_small.jsonl", "w") as f:
        for i, r in enumerate(rows):
            img = Image.open(banana_path(r["path"])).convert("RGB")
            s = rng.uniform(0.20, 0.35) * 256 / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
            bg = texture(rng)
            bg.paste(img, (rng.randint(0, 256 - img.width), rng.randint(0, 256 - img.height)))
            p = out / f"{i:04d}.png"
            bg.save(p)
            f.write(json.dumps(dict(path=str(p), label=r["label"], source="val_small_" + r.get("source", "bananaimagebd"))) + "\n")
    print("val_small", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Build the round 30 banana split file: BananaImageBD (old splits unchanged) + BananaID + Prata Catarina.

usage: build_banana_v7.py OLD_SPLITS_CSV OUT_CSV
  data root: BANANA_EXTRA_DIR (default <repo>/data/banana_extra) with bananaid/ and prata/ extracted inside;
  Prata crops are written to <data root>/crops/prata/.

- BananaImageBD keeps its round 29 train / val / test rows, so the old test of 80 stays comparable.
- BananaID (Mendeley h6n5srjjyw, CC BY 4.0): Ambon is left out (its peel stays green when ripe or
  overripe, so its stages do not follow peel colour). The 4 views of one banana share an index and are
  kept in one split.
- Prata Catarina (Mendeley 7vb4djkbrc, CC BY 4.0): one box per photo, cut with a 10% margin and the
  long side reduced to 768 px. Degrees map to stages: 1-2 unripe, 3-5 turning, 6-7 ripe, 8 overripe
  (degree 7, yellow with brown spots, matches BananaImageBD ripe, not its mostly black overripe).
  Runs of the same stage taken 5 s or less apart are kept in one split (likely the same banana).
- New sources are split 70 / 15 / 15 by group, per stage, with a fixed seed.
"""

import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ.get("BANANA_EXTRA_DIR", Path(__file__).resolve().parents[2] / "data" / "banana_extra"))
BID = ROOT / "bananaid" / "Banana Ripeness Classification Dataset"
BID_MAP = {"Unripe": "unripe", "Half_Ripe": "turning", "Ripe": "ripe", "Overripe": "overripe"}
PRATA_MAP = {1: "unripe", 2: "unripe", 3: "turning", 4: "turning", 5: "turning", 6: "ripe", 7: "ripe", 8: "overripe"}
CROP_DIR = ROOT / "crops" / "prata"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def split_groups(groups, seed):
    """groups: {key: (label, [paths])} -> {path: split}, 70/15/15 per label."""
    by_label = defaultdict(list)
    for k, (lab, _) in groups.items():
        by_label[lab].append(k)
    out = {}
    rng = random.Random(seed)
    for lab in sorted(by_label):
        keys = sorted(by_label[lab])
        rng.shuffle(keys)
        n = len(keys)
        n_te, n_va = round(n * 0.15), round(n * 0.15)
        for i, k in enumerate(keys):
            s = "test" if i < n_te else "val" if i < n_te + n_va else "train"
            for p in groups[k][1]:
                out[p] = s
    return out


def bananaid_rows():
    groups = {}
    for folder, lab in BID_MAP.items():
        for p in sorted((BID / folder).glob("*.jpg")):
            parts = p.stem.split("_")  # Variety_Stage[_Stage]_View_Index
            var, idx = parts[0], parts[-1]
            if var == "Ambon":
                continue
            key = f"{var}_{folder}_{idx}"
            groups.setdefault(key, (lab, []))[1].append(str(p))
    sp = split_groups(groups, seed=30)
    return [(p, lab, sp[p], "bananaid") for lab, ps in groups.values() for p in ps]


def prata_time(name):
    m = re.match(r"(\d{8})_(\d{6})", name)
    if not m:
        return None
    d, t = m.groups()
    return int(d) * 86400 + int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:])


def prata_rows():
    folder = next((ROOT / "prata").glob("*/labeled banana images"))
    j = json.load(open(glob.glob(str(folder / "*.json"))[0]))
    imgs = {i["id"]: i for i in j["images"]}
    anns = defaultdict(list)
    for a in j["annotations"]:
        if a["category_id"] in PRATA_MAP:
            anns[a["image_id"]].append(a)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    items = []  # (time or None, name, crop path, label)
    for iid, al in anns.items():
        info = imgs[iid]
        src = folder / info["file_name"]
        im = None
        for k, a in enumerate(al):
            out = CROP_DIR / f"{Path(info['file_name']).stem}_{k}.jpg"
            if not out.exists():
                if im is None:
                    im = Image.open(src).convert("RGB")
                x, y, w, h = a["bbox"]
                mx, my = 0.1 * w, 0.1 * h
                box = (max(0, x - mx), max(0, y - my), min(im.width, x + w + mx), min(im.height, y + h + my))
                c = im.crop(tuple(int(v) for v in box))
                c.thumbnail((768, 768))
                c.save(out, quality=92)
            items.append((prata_time(info["file_name"]), info["file_name"], str(out), PRATA_MAP[a["category_id"]]))
    # photos come every 5-10 s and the stage often changes between them, so only a run of the same
    # stage with gaps of 5 s or less counts as one banana; names without a time are their own group
    groups = {}
    timed = sorted((t, n, p, lab) for t, n, p, lab in items if t is not None)
    gid, last, last_lab = -1, None, None
    for t, n, p, lab in timed:
        if last is None or t - last > 5 or lab != last_lab:
            gid += 1
        last, last_lab = t, lab
        groups.setdefault(f"t{gid}_{lab}", (lab, []))[1].append(p)
    for t, n, p, lab in items:
        if t is None:
            groups.setdefault(f"n_{n}", (lab, []))[1].append(p)
    sp = split_groups(groups, seed=31)
    return [(p, lab, sp[p], "prata") for lab, ps in groups.values() for p in ps]


def main(old_csv, out_csv):
    rows = []
    for r in csv.DictReader(open(old_csv)):
        rows.append(dict(path=r["path"], label=r["label"], split=r["split"], source="bananaimagebd", sha256=r["sha256"]))
    for p, lab, s, src in bananaid_rows() + prata_rows():
        rows.append(dict(path=p, label=lab, split=s, source=src, sha256=sha(p)))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source", "sha256"])
        w.writeheader()
        w.writerows(rows)
    cnt = defaultdict(int)
    for r in rows:
        cnt[(r["source"], r["split"], r["label"])] += 1
    for k in sorted(cnt):
        print(*k, cnt[k])


if __name__ == "__main__":
    main(*sys.argv[1:3])

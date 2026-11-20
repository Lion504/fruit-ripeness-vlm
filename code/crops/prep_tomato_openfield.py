"""Tomato (OpenField-BD, Zenodo 20176021): YOLO boxes -> ripeness crops, split by capture session / photo group.
The dataset's own train/val/test folders are merged and re-split (they are image-level and ignore sessions)."""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prep_lib as L
from paths import ROOT
from PIL import Image

OUT = Path(ROOT) / "crops/tomato"
SRC = Path(os.environ.get("TOMATO_DATASET_DIR", Path(ROOT) / "data" / "tomato" / "tomato_m_f"))  # OpenField-BD, extracted
CLASSES = ["green", "half-ripe", "fully-ripe"]
ID2CLS = {
    0: "fully-ripe",
    1: "half-ripe",
    2: "green",
}  # from data.yaml: ['full_ripened','half_ripened','green']
SESSION_GAP_S = 30

OUT.mkdir(parents=True, exist_ok=True)
excl = Counter()


def stamp(name):
    m = re.search(r"IMG_?(\d{8})_?(\d{6})", name)
    return (
        datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        if m
        else None
    )


imgs = sorted(
    p for s in ["train", "val", "test"] for p in (SRC / s / "images").glob("*.jpg")
)
anns = {}
for p in imgs:
    W, H = Image.open(p).size
    lab = p.parent.parent / "labels" / (p.stem + ".txt")
    for line in lab.read_text().split("\n"):
        t = line.split()
        if len(t) != 5:
            if t:
                excl["non-box label line"] += 1
            continue
        c, cx, cy, bw, bh = int(t[0]), *map(float, t[1:])
        b = ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)
        w, h = b[2] - b[0], b[3] - b[1]
        if min(w, h) < L.MIN_SIDE:
            excl[f"short side < {L.MIN_SIDE}px"] += 1
            continue
        touch = b[0] <= 2 or b[1] <= 2 or b[2] >= W - 3 or b[3] >= H - 3
        if touch and (min(w, h) < 48 or max(w, h) / min(w, h) > 1.6):
            excl["cut by image border (small or elongated)"] += 1
            continue
        anns.setdefault(p, []).append((ID2CLS[c], tuple(round(v, 1) for v in b), touch))

paths = sorted(anns)
ts = sorted({(stamp(p.name), p) for p in paths if stamp(p.name)})
key_of, sid, last = {}, 0, None
for t, p in ts:
    if last is not None and t - last > SESSION_GAP_S:
        sid += 1
    key_of[p], last = f"s{sid}", t
keys = [key_of.get(p) for p in paths]
gid, gstats = L.group_sources(paths, extra_keys=keys)
gcls = defaultdict(Counter)
for p in paths:
    for cls, b, t in anns[p]:
        gcls[gid[p]][cls] += 1
gsplit = L.split_groups(gcls, seed=0)

rows = []
for p in paths:
    sp = gsplit[gid[p]]
    with Image.open(p) as im:
        im.load()
        for j, (cls, b, t) in enumerate(anns[p]):
            c = L.crop_fruit(im, b)
            rel = f"{sp}/{cls}/{p.stem}_{j:02d}.jpg"
            (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
            c.save(OUT / rel, quality=92)
            rows.append(
                dict(
                    crop_path=rel,
                    source_image=str(p.relative_to(SRC.parent.parent)),
                    **{"class": cls},
                    box=",".join(str(round(v)) for v in b),
                    width=c.width,
                    height=c.height,
                    split=sp,
                    group=gid[p],
                    truncated=int(t),
                )
            )
L.write_csv(rows, OUT / "crops.csv")
for s in L.SPLIT_FRAC:
    L.write_csv([r for r in rows if r["split"] == s], OUT / f"{s}.csv")
L.contact_sheet(rows, OUT, OUT / "contact_sheet_1.jpg", seed=1)
L.contact_sheet(rows, OUT, OUT / "contact_sheet_2.jpg", seed=2)
g2s = defaultdict(set)
for r in rows:
    g2s[r["group"]].add(r["split"])
rep = {
    "groups": gstats,
    "excluded": dict(excl),
    "kept": len(rows),
    "sources_with_crops": len(paths),
    "groups_spanning_splits": sum(len(v) > 1 for v in g2s.values()),
}
print(json.dumps(rep, indent=1))
print(L.counts_table(rows, CLASSES))
json.dump(rep, open(OUT / "report.json", "w"), indent=1)

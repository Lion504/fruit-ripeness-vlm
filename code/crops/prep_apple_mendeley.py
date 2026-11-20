"""Apple (Mendeley gfcmdbvw65): VIA polygon annotations -> stage crops, split by source photo group.
Only original photos are used; the dataset's *_brightness / *_noise copies and the *_resize folders are ignored."""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prep_lib as L
from paths import ROOT
from PIL import Image

OUT = Path(ROOT) / "crops/apple"
SRC = Path(os.environ.get("APPLE_DATASET_DIR", Path(ROOT) / "data" / "apple"))  # the extracted Mendeley gfcmdbvw65 v4 folder
CLASSES = ["immature", "semi-mature", "mature"]
MAP = {
    "immature apple": "immature",
    "semi-mature apple": "semi-mature",
    "mature apple": "mature",
}

OUT.mkdir(parents=True, exist_ok=True)
excl = Counter()
items = []  # (img_path, cls, box, truncated)
for orig_split in ["train", "val", "test"]:
    d = json.load(open(SRC / f"{orig_split}.json"))
    for v in d.values():
        fn = v["filename"]
        if re.search(r"_(brightness|noise)\.jpg$", fn):
            excl["dataset augmented copy (region)"] += len(v["regions"])
            continue
        p = SRC / orig_split / fn
        for r in v["regions"]:
            sa, ra = r["shape_attributes"], r["region_attributes"]
            cls = MAP.get(ra.get("name"))
            if cls is None or "all_points_x" not in sa:
                excl["no/unknown label"] += 1
                continue
            x, y = sa["all_points_x"], sa["all_points_y"]
            items.append((p, cls, (min(x), min(y), max(x), max(y))))

by_img = defaultdict(list)
for it in items:
    by_img[it[0]].append(it)
keep = []
sizes = {}
for p, its in by_img.items():
    with Image.open(p) as im:
        cb = L.content_box(im)
    sizes[p] = cb
    for _, cls, b in its:
        b = (max(b[0], cb[0]), max(b[1], cb[1]), min(b[2], cb[2]), min(b[3], cb[3]))
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 0 or h <= 0:
            excl["empty after clipping to content"] += 1
            continue
        if min(w, h) < L.MIN_SIDE:
            excl[f"short side < {L.MIN_SIDE}px"] += 1
            continue
        touch = (
            b[0] <= cb[0] + 1
            or b[1] <= cb[1] + 1
            or b[2] >= cb[2] - 2
            or b[3] >= cb[3] - 2
        )
        if touch and (min(w, h) < 48 or max(w, h) / min(w, h) > 1.6):
            excl["cut by image border (small or elongated)"] += 1
            continue
        keep.append((p, cls, b, touch))

paths = sorted({k[0] for k in keep})


def session_key(p):
    # camera-roll numbers (IMG_nnnn) within 2 of each other, chained, count as one capture session
    m = re.match(r"IMG_(\d+)", p.name)
    return m and int(m.group(1))


nums = sorted({session_key(p) for p in paths if session_key(p)})
chain, cid = {}, 0
for i, n in enumerate(nums):
    if i and n - nums[i - 1] > 2:
        cid += 1
    chain[n] = cid
keys = [f"s{chain[session_key(p)]}" if session_key(p) else None for p in paths]
gid, gstats = L.group_sources(paths, extra_keys=keys)
gcls = defaultdict(Counter)
for p, cls, b, t in keep:
    gcls[gid[p]][cls] += 1
gsplit = L.split_groups(gcls, seed=0)

rows = []
for i, (p, cls, b, t) in enumerate(sorted(keep, key=lambda k: (str(k[0]), k[2]))):
    sp = gsplit[gid[p]]
    with Image.open(p) as im:
        cb = sizes[p]
        c = L.crop_fruit(
            im.crop(cb), (b[0] - cb[0], b[1] - cb[1], b[2] - cb[0], b[3] - cb[1])
        )
    rel = f"{sp}/{cls}/{p.stem}_{i:05d}.jpg"
    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
    c.save(OUT / rel, quality=92)
    rows.append(
        dict(
            crop_path=rel,
            source_image=str(p.relative_to(SRC.parent)),
            **{"class": cls},
            box=",".join(map(str, b)),
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
# group leak check
g2s = defaultdict(set)
for r in rows:
    g2s[r["group"]].add(r["split"])
src2s = defaultdict(set)
for r in rows:
    src2s[r["source_image"]].add(r["split"])
rep = {
    "groups": gstats,
    "excluded": dict(excl),
    "kept": len(rows),
    "groups_spanning_splits": sum(len(v) > 1 for v in g2s.values()),
    "sources_spanning_splits": sum(len(v) > 1 for v in src2s.values()),
    "size_stats": {
        s: sorted(int(min(r["width"], r["height"])) for r in rows if r["split"] == s)[
            len([1 for r in rows if r["split"] == s]) // 2
        ]
        for s in L.SPLIT_FRAC
    },
}
print(json.dumps(rep, indent=1))
print(L.counts_table(rows, CLASSES))
json.dump(rep, open(OUT / "report.json", "w"), indent=1)

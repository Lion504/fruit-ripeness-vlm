"""Shared helpers for building crop classification sets from detection/segmentation datasets.

Steps: crop each annotated fruit -> group source photos (exact + near duplicates) ->
split by group, stratified by class -> write CSV and contact sheets.
"""

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw

MARGIN = 0.10
MIN_SIDE = 32
MAX_SIDE = 256
PHASH_DIST = 6  # hamming distance (of 64 bit pHash) treated as near duplicate
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_box(im, thr=12):
    """Bounding box of non-black content (some sources are letterboxed with black bars)."""
    import numpy as np

    a = np.asarray(im.convert("L"))
    cols = np.where(a.max(axis=0) > thr)[0]
    rows = np.where(a.max(axis=1) > thr)[0]
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, im.width, im.height)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def crop_fruit(im, box, margin=MARGIN, max_side=MAX_SIDE):
    """box = (x0, y0, x1, y1) in pixels. Returns RGB crop with margin, downscaled to max_side."""
    W, H = im.size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    mx, my = w * margin, h * margin
    c = im.crop(
        (
            max(0, int(x0 - mx)),
            max(0, int(y0 - my)),
            min(W, int(round(x1 + mx))),
            min(H, int(round(y1 + my))),
        )
    )
    c = c.convert("RGB")
    s = max(c.size)
    if s > max_side:
        k = max_side / s
        c = c.resize(
            (max(1, round(c.width * k)), max(1, round(c.height * k))), Image.LANCZOS
        )
    return c


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def group_sources(paths, dist=PHASH_DIST, extra_keys=None):
    """Cluster source images: identical sha256, pHash within `dist`, or same extra key.
    Returns (group_id per path dict, stats dict)."""
    n = len(paths)
    uf = UF(n)
    sha = {}
    exact = 0
    hashes = []
    for i, p in enumerate(paths):
        s = sha256(p)
        if s in sha:
            uf.union(i, sha[s])
            exact += 1
        else:
            sha[s] = i
        with Image.open(p) as im:
            im.draft("RGB", (512, 512))  # fast decode for large JPEGs
            hashes.append(imagehash.phash(im.convert("RGB")))
    near = 0
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= dist and uf.find(i) != uf.find(j):
                uf.union(i, j)
                near += 1
    if extra_keys:
        first = {}
        for i, k in enumerate(extra_keys):
            if k is None:
                continue
            if k in first:
                uf.union(i, first[k])
            else:
                first[k] = i
    gid = {paths[i]: uf.find(i) for i in range(n)}
    stats = {
        "sources": n,
        "exact_dup_files": exact,
        "near_dup_links": near,
        "groups": len({uf.find(i) for i in range(n)}),
    }
    return gid, stats


def split_groups(group_classes, seed=0, frac=SPLIT_FRAC):
    """group_classes: {gid: Counter(class->n crops)}. Greedy stratified assignment of whole groups.
    Returns {gid: split}."""
    rng = random.Random(seed)
    total = Counter()
    for c in group_classes.values():
        total.update(c)
    target = {s: {k: v * f for k, v in total.items()} for s, f in frac.items()}
    cur = {s: Counter() for s in frac}
    gids = list(group_classes)
    rng.shuffle(gids)
    gids.sort(
        key=lambda g: -sum(group_classes[g].values())
    )  # large groups first (stable after shuffle)
    out = {}
    for g in gids:
        best, best_cost = None, None
        for s in frac:
            cost = 0.0
            for k, tgt in target[s].items():
                new = cur[s][k] + group_classes[g].get(k, 0)
                cost += ((new - tgt) / max(tgt, 1.0)) ** 2 - (
                    (cur[s][k] - tgt) / max(tgt, 1.0)
                ) ** 2
            cost += rng.random() * 1e-6
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        out[g] = best
        cur[best].update(group_classes[g])
    return out


def write_csv(rows, path):
    cols = [
        "crop_path",
        "source_image",
        "class",
        "box",
        "width",
        "height",
        "split",
        "group",
        "truncated",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def counts_table(rows, classes):
    t = {s: Counter() for s in SPLIT_FRAC}
    for r in rows:
        t[r["split"]][r["class"]] += 1
    lines = [
        "| split | " + " | ".join(classes) + " | total |",
        "|---|" + "---|" * (len(classes) + 1),
    ]
    for s in SPLIT_FRAC:
        lines.append(
            f"| {s} | "
            + " | ".join(str(t[s][c]) for c in classes)
            + f" | {sum(t[s].values())} |"
        )
    return "\n".join(lines)


def contact_sheet(rows, root, out_path, n=40, seed=0, cols=8, cell=160, classes=None):
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    nr = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, nr * (cell + 16)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        im = Image.open(Path(root) / r["crop_path"]).convert("RGB")
        im.thumbnail((cell - 4, cell - 4))
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        sheet.paste(im, (x + 2, y + 16))
        d.text(
            (x + 3, y + 2),
            f"{r['class']} {r['width']}x{r['height']}",
            fill=(255, 255, 0),
        )
    sheet.save(out_path, quality=90)

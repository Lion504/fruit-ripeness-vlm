"""Build the round 32 extra tomato train list from ParsaVictor's ripeness crops (CC BY 4.0).

usage: build_tomato_parsa_train.py CROPS_ZIP OUT_DIR [N_RAW N_HALF N_RIPE [PER_BOX SEED]]
       (default 900 600 1000 0 32 = round 32; round 36: 2000 2000 2000 1 36)
PER_BOX=1: at most one crop per tomato (photo + box index) instead of one per photo.

- Only their TRAIN split is used; rotten is left out. Labels: raw and half_ripe -> not_yet, ripe -> ready
  (the same mapping used to score their test).
- Their train split holds Roboflow copies of one photo (rotated, flipped). Crops are grouped by source photo
  (name before the '_<ext>.rf.' part, with rotation / flip tags removed) and at most one crop per photo is taken.
- Near-copy check: a 64-bit difference hash of every picked crop is compared with every crop of the tomato
  val and test, ParsaVictor's test, and the 300-crop ParsaVictor val sample. Picked crops within 6 bits of any
  of them are dropped and counted.
Writes OUT_DIR/<class>/<name>.jpg, OUT_DIR/extra_train.csv (path,label,their_class) and OUT_DIR/build_report.json.
"""

import csv
import json
import os
import random
import re
import sys
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_tune"))
from paths import CROPS  # noqa: E402

MAP = {"raw": "not_yet", "half_ripe": "not_yet", "ripe": "ready"}
# PARSAVICTOR_DIR holds the extracted test crops (test/<class>/) and the validation samples made by
# build_parsa_val.py (val300/<class>/, val1000/<class>/)
PARSA = Path(os.environ.get("PARSAVICTOR_DIR", Path(__file__).resolve().parents[2] / "data" / "parsavictor"))


def dhash(im):
    g = im.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def source_key(name):
    stem = Path(name).stem
    stem = re.split(r"_(?:png|jpg|jpeg|JPG|PNG|JPEG)\.rf\.", stem)[0]
    stem = re.sub(r"_(rotated_\d+|flipped_[hv])", "", stem)
    return stem


def main(zip_path, out_dir, n_raw=900, n_half=600, n_ripe=1000, per_box="0", seed="32"):
    out = Path(out_dir).resolve()
    z = zipfile.ZipFile(zip_path)
    want = {"raw": int(n_raw), "half_ripe": int(n_half), "ripe": int(n_ripe)}
    groups = {c: defaultdict(list) for c in want}
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] == "train" and parts[2] in want and parts[3]:
            key = source_key(parts[3])
            if per_box == "1":  # round 36: one crop per tomato (box) instead of one per photo
                key += "#" + Path(parts[3]).stem.rsplit("_", 1)[-1]
            groups[parts[2]][key].append(n)

    ref = []
    for split in ("val", "test"):
        for r in csv.DictReader(open(CROPS / "tomato" / f"{split}.csv")):
            ref.append(("ours_" + split, CROPS / "tomato" / r["crop_path"]))
    ref += [("parsa_test", p) for p in sorted((PARSA / "test").glob("*/*"))]
    ref += [("parsa_val300", p) for p in sorted((PARSA / "val300").glob("*/*"))]
    ref += [("parsa_val1000", p) for p in sorted((PARSA / "val1000").glob("*/*"))]
    ref_h = [(s, dhash(Image.open(p))) for s, p in ref]

    rng = random.Random(int(seed))
    rows, report = [], {"groups": {c: len(g) for c, g in groups.items()}, "dropped": defaultdict(int), "kept": {}}
    for c, g in groups.items():
        keys = sorted(g)
        rng.shuffle(keys)
        kept = 0
        for k in keys:
            if kept >= want[c]:
                break
            n = rng.choice(sorted(g[k]))
            im = Image.open(BytesIO(z.read(n))).convert("RGB")
            h = dhash(im)
            hit = next((s for s, rh in ref_h if bin(h ^ rh).count("1") <= 6), None)
            if hit:
                report["dropped"][f"{c}~{hit}"] += 1
                continue
            p = out / c / Path(n).name
            p.parent.mkdir(parents=True, exist_ok=True)
            im.save(p, quality=95)
            rows.append(dict(path=str(p), label=MAP[c], their_class=c))
            kept += 1
        report["kept"][c] = kept
    with open(out / "extra_train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "their_class"])
        w.writeheader()
        w.writerows(rows)
    report["reference_crops"] = len(ref_h)
    json.dump(report, open(out / "build_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

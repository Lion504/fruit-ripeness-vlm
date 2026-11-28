"""Export the new non-fruit test list (round 39) with paths relative to each public source dataset.

usage: export_nonfruit_r39_list.py MANIFEST IMG_ROOT RESULTS_NEW RESULTS_PREV OUT_CSV [SPLIT]
MANIFEST      manifest_r39.csv written by code/data_prep/build_neg_r39.py
IMG_ROOT      the folder with the built photos (IMG_ROOT/<split>/<name>.jpg), used for the SHA-256
RESULTS_NEW   results_*.jsonl of the fruit-naming test with the retrained add-on (rows of the set neg_r39_test)
RESULTS_PREV  the same for the add-on without the extra non-fruit photos
SPLIT         test (default) or val
Writes one row per photo: dataset, file (inside the source dataset), box (the object box before the 10% margin),
label (always none), SHA-256 of the built photo, box unit, category, licence of the photo, name of the built file, and
the answer of both add-ons. No absolute paths are written. The SHA-256 is of the built JPEG (cropped, long side at most
512 px, quality 92); the same bytes need the same Pillow version, so on a mismatch compare the photos by eye or by hash
of the decoded pixels.
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


def licence_name(url):
    m = re.search(r"/licenses/([a-z-]+)/(\d\.\d)", url)
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else url


def source_file(r):
    sid = r["source_id"]
    if r["source"] == "COCO2017val":
        return "COCO 2017", f"val2017/{int(sid):012d}.jpg", "pixels"
    if r["source"].startswith("OpenImages"):
        return "Open Images V7", f"{r['source'][len('OpenImages'):]}/{sid}.jpg", "fraction of width and height"
    return "Fruits-360 100x100", sid, ""


def answers(path, group="neg_r39_test"):
    out = {}
    for line in open(path):
        d = json.loads(line)
        if d["set"] == group:
            out[Path(d["path"]).name] = d["pred"]
    return out


def main(manifest, img_root, res_new, res_prev, out_csv, split="test"):
    new, prev = answers(res_new), answers(res_prev)
    rows = []
    for r in csv.DictReader(open(manifest)):
        if r["split"] != split:
            continue
        name = Path(r["path"]).name
        ds, f, unit = source_file(r)
        box = ",".join(str(round(float(v), 4)) for v in json.loads(r["box"])) if r["box"] else ""
        sha = hashlib.sha256(open(Path(img_root) / split / name, "rb").read()).hexdigest()
        rows.append(dict(dataset=ds, file=f, box=box, label="none", sha256=sha, box_unit=unit if box else "",
                         category=r["category"], licence=licence_name(r["licence"]), name=name,
                         answer_round39=new.get(name, ""), answer_round33=prev.get(name, "")))
    rows.sort(key=lambda d: d["name"])
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "photos written")


if __name__ == "__main__":
    main(*sys.argv[1:])

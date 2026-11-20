"""Export the test lists and per-photo answers with paths relative to each public dataset.

usage: export_test_lists.py E2E_RESULTS.jsonl OUT_DIR
Reads the banana split file (BANANA_SPLITS), the apple / tomato crop lists (CROPS/<fruit>/test.csv), the
ParsaVictor test list and the robustness lists, and the two-pass answers (two_pass_v7.py results.jsonl).
Writes OUT_DIR/test_<set>.csv (dataset, file, box, label, sha256 of the photo scored where known) and
OUT_DIR/answers_two_pass.csv (set, dataset, file, box, label, pass1, stage) and OUT_DIR/test_negatives.csv
(the photos without a supported fruit, which are not published: number, description, expected answer, SHA256).
No absolute paths are written.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import os

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "qwen_tune"))
from paths import CROPS, SPLITS_CSV, banana_path  # noqa: E402

LISTS = Path(os.environ.get("TEST_LISTS_DIR", REPO / "data" / "lists"))  # frame_small, frame_cut, bg_new2, parsa_test .jsonl
BIBD = "Banana Ripeness Detection Dataset/"
BID = "Banana Ripeness Classification Dataset/"
TOMATO_MAP = {"green": "not_yet", "half-ripe": "not_yet", "fully-ripe": "ready"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(path, source):
    """(dataset, file, box) for one scored photo."""
    p = str(path)
    if source == "bananaimagebd" or BIBD in p:
        return "BananaImageBD v2", p[p.index(BIBD):], ""
    if source == "bananaid" or BID in p:
        return "BananaID (Mendeley h6n5srjjyw)", p[p.index(BID):], ""
    if source == "prata" or "/crops/prata/" in p:
        stem, k = Path(p).stem.rsplit("_", 1)
        return "Banana Prata Catarina (Mendeley 7vb4djkbrc)", stem + ".jpg", f"annotation {k} (10% margin)"
    if "/cls_tomato/" in p:
        return "ParsaVictor tomato_ripeness_crops.zip", "cls_tomato/" + p.split("/cls_tomato/")[1], ""
    if "/hard_r29/" in p:
        return "robustness set (derived from BananaImageBD v2)", p.split("/hard_r29/")[1], ""
    raise ValueError(p)


def crop_index(fruit):
    out = {}
    for r in csv.DictReader(open(CROPS / fruit / "test.csv")):
        ds = "Multi-Stage Apple Dataset (Mendeley gfcmdbvw65 v4)" if fruit == "apple" else \
            "OpenField-BD Tomato Maturity (Zenodo 20176021)"
        src = r["source_image"].split("/", 1)[1] if fruit == "apple" else r["source_image"].split("/", 1)[1]
        lab = TOMATO_MAP[r["class"]] if fruit == "tomato" else r["class"]
        out[str(CROPS / fruit / r["crop_path"])] = (ds, src, r["box"], lab)
    return out


def main(e2e, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops = {**crop_index("apple"), **crop_index("tomato")}
    sets = {}
    for r in csv.DictReader(open(SPLITS_CSV)):
        if r["split"] == "test":
            p = banana_path(r["path"])
            ds, f, box = rel(p, r["source"])
            sets.setdefault("banana", []).append(dict(dataset=ds, file=f, box=box, label=r["label"], sha256=r["sha256"]))
    for name in ("frame_small", "frame_cut", "bg_new2"):
        for line in open(LISTS / f"{name}.jsonl"):
            d = json.loads(line)
            ds, f, box = rel(d["path"], name)
            sets.setdefault(name, []).append(dict(dataset=ds, file=f, box=box, label=d["label"], sha256=sha(d["path"])))
    for p, (ds, f, box, lab) in crops.items():
        fruit = "apple" if "Apple" in ds else "tomato"
        sets.setdefault(fruit, []).append(dict(dataset=ds, file=f, box=box, label=lab, sha256=sha(p)))
    for line in open(LISTS / "parsa_test.jsonl"):
        d = json.loads(line)
        ds, f, box = rel(d["path"], "parsa")
        sets.setdefault("tomato_parsavictor", []).append(dict(dataset=ds, file=f, box=box, label=d["label"], sha256=sha(d["path"])))
    for name, rows in sets.items():
        with open(out / f"test_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["dataset", "file", "box", "label", "sha256"])
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows))
    neg_rows, k = [], 0
    with open(out / "answers_two_pass.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "dataset", "file", "box", "label", "pass1", "stage"])
        w.writeheader()
        n = 0
        for line in open(e2e):
            d = json.loads(line)
            if d["set"] == "negatives":
                # photos without a supported fruit are not published: numbered, with a short description and SHA256
                k += 1
                name = Path(d["path"]).name
                desc = ("darkened or blurred photo (expected: unclear)" if name.startswith("dark")
                        else "background patch without fruit" if name.startswith("bg_")
                        else "landscape photo without fruit")
                ds, f, box = "photos without a supported fruit (not published)", f"neg_{k:03d}", ""
                neg_rows.append(dict(file=f, description=desc, expected=d["fruit"], sha256=sha(d["path"])))
            elif d["path"] in crops:
                ds, f, box, _ = crops[d["path"]]
            else:
                ds, f, box = rel(d["path"], d.get("source", ""))
            w.writerow(dict(set=d["set"], dataset=ds, file=f, box=box, label=d.get("label") or d["fruit"],
                            pass1=d.get("pass1"), stage=d.get("stage_routed")))
            n += 1
        print("answers", n)
    with open(out / "test_negatives.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "description", "expected", "sha256"])
        w.writeheader()
        w.writerows(neg_rows)
    print("negatives", len(neg_rows))


if __name__ == "__main__":
    main(*sys.argv[1:3])

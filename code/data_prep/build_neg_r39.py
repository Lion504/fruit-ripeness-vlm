"""Round 39: build the non-fruit ("none") photo set for the fruit-naming add-on.

usage: build_neg_r39.py ROOT        ROOT holds dl/ (COCO zips), oi/ (Open Images csv files), f360/f360.zip
Writes ROOT/img/<split>/<name>.jpg (long side <= 512) and ROOT/manifest_r39.csv
(columns: split,path,category,group,source,source_id,licence,author,box).

Sources (all open, official hosts): COCO 2017 val (image licences per photo from the dataset's own list, photos with a
NoDerivs licence are skipped because they are cropped; the COCO annotations are CC BY 4.0, COCO terms
of use), Open Images validation and test sets (only photos whose own licence is CC BY 2.0; annotations CC BY 4.0),
Fruits-360 100x100 (CC BY-SA 4.0; split by fruit variant folder).
Rules: no photo that carries an apple, banana or tomato label (or a food scene where tomatoes are likely) is used; all
crops of one source photo go to one split (hash of the photo id); crops use the object box plus a 10% margin.
"""

import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(sys.argv[1])
OUT = ROOT / "img"
rnd = random.Random(39)
SPLIT_SHARE = (("train", 0.72), ("val", 0.14), ("test", 0.14))


def split_of(key):
    h = int(hashlib.sha256(("r39:" + key).encode()).hexdigest(), 16) % 10000 / 10000
    acc = 0
    for s, p in SPLIT_SHARE:
        acc += p
        if h < acc:
            return s
    return "test"


def crop_box(img, x0, y0, x1, y1, margin=0.10):
    w, h = img.size
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0, x0 - bw * margin), min(w, x1 + bw * margin)
    y0, y1 = max(0, y0 - bh * margin), min(h, y1 + bh * margin)
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def shrink(img, side=512):
    s = side / max(img.size)
    if s < 1:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return img


# group of each category (for the quota)
GROUP = {}
for g, names in {
    "produce": "orange broccoli carrot lemon pear peach bell_pepper potato onion cucumber grape strawberry mango pineapple pumpkin grapefruit plum cabbage radish garlic zucchini",
    "round": "sports_ball ball",
    "electronics": "laptop cell_phone keyboard mouse remote tv mobile_phone computer_keyboard computer_monitor computer_mouse remote_control tablet_computer headphones television toaster microwave",
    "household": "cup bottle wine_glass chair couch bed backpack handbag suitcase umbrella tie book clock vase scissors toothbrush mug coffee_cup potted_plant bench oven refrigerator sink",
    "shoe_toy": "footwear toy teddy_bear kite frisbee",
    "people": "person human_face human_hand",
    "animal_car": "cat dog bird horse car",
    "scene": "scene",
    "produce_f360": "f360",
}.items():
    for n in names.split():
        GROUP[n] = g

records = []  # dicts: key, split, category, source, source_id, licence, author, box, getimg()


# ---------------------------------------------------------------- COCO
def coco():
    z = zipfile.ZipFile(ROOT / "dl" / "annotations_trainval2017.zip")
    d = json.load(z.open("annotations/instances_val2017.json"))
    cats = {c["id"]: c["name"].replace(" ", "_") for c in d["categories"]}
    lic = {l["id"]: l["url"] for l in d["licenses"]}
    img = {i["id"]: i for i in d["images"]}
    bad = {"apple", "banana", "sandwich", "hot_dog", "pizza", "cake", "donut", "bowl", "dining_table"}
    badimg = {a["image_id"] for a in d["annotations"] if cats[a["category_id"]] in bad}
    use = {"orange", "broccoli", "carrot", "sports_ball", "laptop", "cell_phone", "keyboard", "mouse", "remote", "tv",
           "cup", "bottle", "wine_glass", "chair", "couch", "bed", "backpack", "handbag", "suitcase", "umbrella", "tie",
           "book", "clock", "vase", "scissors", "toothbrush", "teddy_bear", "kite", "frisbee", "person", "cat", "dog",
           "bird", "horse", "car", "toaster", "microwave", "oven", "refrigerator", "sink", "potted_plant", "bench"}
    zi = zipfile.ZipFile(ROOT / "dl" / "val2017.zip")
    n_skip_lic = 0
    by_img = {}
    for a in d["annotations"]:
        c = cats[a["category_id"]]
        if a["image_id"] in badimg or c not in use or a.get("iscrowd"):
            continue
        im = img[a["image_id"]]
        if im["license"] in (3, 6):
            n_skip_lic += 1
            continue
        x, y, w, h = a["bbox"]
        if w * h < 0.10 * im["width"] * im["height"] and not (c in ("orange", "sports_ball") and w * h > 0.02 * im["width"] * im["height"]):
            continue
        if min(w, h) < 80:
            continue
        by_img.setdefault(a["image_id"], {}).setdefault(c, []).append((w * h, (x, y, x + w, y + h)))
    out = []
    for iid, cd in by_img.items():
        im = img[iid]
        # one crop per category per photo, at most 2 crops per photo
        picks = sorted(cd.items(), key=lambda kv: rnd.random())[:2]
        for c, lst in picks:
            box = max(lst)[1]
            out.append(dict(key=f"coco{iid}_{c}", pid=f"coco{iid}", category=c, source="COCO2017val", source_id=str(iid),
                            licence=lic[im["license"]], author="Flickr user (see COCO flickr_url)", box=box,
                            loader=lambda zi=zi, f=im["file_name"]: Image.open(io.BytesIO(zi.read("val2017/" + f))).convert("RGB")))
    # whole scenes: photos without any excluded category, chosen later
    for iid, im in img.items():
        if iid in badimg or im["license"] in (3, 6):
            continue
        out.append(dict(key=f"coco{iid}_scene", pid=f"coco{iid}", category="scene", source="COCO2017val", source_id=str(iid),
                        licence=lic[im["license"]], author="Flickr user (see COCO flickr_url)", box=None,
                        loader=lambda zi=zi, f=im["file_name"]: Image.open(io.BytesIO(zi.read("val2017/" + f))).convert("RGB")))
    return out, n_skip_lic, len(badimg)


# ---------------------------------------------------------------- Open Images
OI_CLASSES = ["Footwear", "Human hand", "Human face", "Toy", "Ball", "Orange", "Lemon", "Pear", "Peach", "Bell pepper",
              "Potato", "Onion", "Cucumber", "Grape", "Strawberry", "Mango", "Pineapple", "Carrot", "Pumpkin", "Grapefruit",
              "Plum", "Cabbage", "Radish", "Garlic", "Zucchini", "Laptop", "Mobile phone", "Computer keyboard",
              "Computer monitor", "Computer mouse", "Remote control", "Tablet computer", "Headphones", "Television", "Mug",
              "Coffee cup", "Bottle", "Cat", "Dog", "Car", "Chair", "Couch", "Bed", "Backpack", "Handbag", "Clock", "Vase",
              "Book", "Toaster", "Microwave oven"]
OI_BAD = ["Apple", "Banana", "Tomato", "Salad", "Pizza", "Sandwich", "Hot dog", "Cake"]


SMALL_OK = {"Orange", "Lemon", "Pear", "Peach", "Bell pepper", "Potato", "Onion", "Cucumber", "Grape", "Strawberry",
            "Mango", "Pineapple", "Carrot", "Pumpkin", "Grapefruit", "Plum", "Ball", "Garlic", "Radish", "Zucchini"}


def oi():
    names = {}
    for r in csv.reader(open(ROOT / "oi" / "oidv7-class-descriptions.csv")):
        names[r[1]] = r[0]
    alias = {"Ball": "Ball (Object)", "Orange": "Orange (fruit)", "Lemon": "Lemon (plant)", "Plum": "Plum (Fruit)"}
    missing = [c for c in OI_CLASSES if alias.get(c, c) not in names]
    print("Open Images classes not found:", missing)
    want = {names[alias.get(c, c)]: c for c in OI_CLASSES if c not in missing}
    bad = {names[c] for c in OI_BAD}
    out, nbad = [], 0
    for sub in ("validation", "test"):
        badimg = set()
        for r in csv.DictReader(open(ROOT / "oi" / f"{sub}-annotations-human-imagelabels.csv")):
            if r["LabelName"] in bad and r["Confidence"] == "1":
                badimg.add(r["ImageID"])
        boxes = {}
        for r in csv.DictReader(open(ROOT / "oi" / f"{sub}-annotations-bbox.csv")):
            if r["LabelName"] in bad:
                badimg.add(r["ImageID"])
            elif r["LabelName"] in want and r["IsGroupOf"] == "0" and r["IsDepiction"] != "1":
                x0, x1, y0, y1 = (float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
                small = want[r["LabelName"]] in SMALL_OK
                if (x1 - x0) * (y1 - y0) >= (0.05 if small else 0.12) and min(x1 - x0, y1 - y0) > (0.15 if small else 0.25):
                    boxes.setdefault(r["ImageID"], []).append((want[r["LabelName"]], (x0, y0, x1, y1)))
        meta = {}
        for r in csv.DictReader(open(ROOT / "oi" / f"{sub}-images-with-rotation.csv")):
            if r["ImageID"] in boxes and r["ImageID"] not in badimg and r["License"].endswith("/by/2.0/") and not r["Rotation"].strip("0."):
                meta[r["ImageID"]] = r
        nbad += len(badimg)
        for iid, bl in boxes.items():
            if iid not in meta:
                continue
            seen = set()
            for c, b in sorted(bl, key=lambda t: rnd.random()):
                if c in seen or len(seen) >= 2:
                    continue
                seen.add(c)
                cat = c.lower().replace(" ", "_")
                out.append(dict(key=f"oi{iid}_{cat}", pid=f"oi{iid}", category=cat, source="OpenImages" + sub, source_id=iid,
                                licence=meta[iid]["License"], author=meta[iid]["Author"], box=b, rel=True,
                                url=f"https://open-images-dataset.s3.amazonaws.com/{sub}/{iid}.jpg"))
    return out, nbad


# ---------------------------------------------------------------- Fruits-360
def f360():
    z = zipfile.ZipFile(ROOT / "f360" / "f360.zip")
    ban = ("apple", "banana", "tomato")
    cls = {}
    for n in z.namelist():
        if n.endswith(".jpg"):
            parts = n.split("/")
            if len(parts) >= 4 and parts[-3] == "Training":
                cls.setdefault(parts[-2], []).append(n)
    keep = {c: v for c, v in cls.items() if not any(b in c.lower().replace("pineapple", "") for b in ban)}
    out = []
    for c, files in keep.items():
        sp = split_of("f360:" + c)
        for n in rnd.sample(sorted(files), min(4, len(files))):
            out.append(dict(key="f360_" + Path(n).stem, pid="f360:" + c, category="f360", source="Fruits-360-100x100",
                            source_id=n.split("/", 1)[1], licence="https://creativecommons.org/licenses/by-sa/4.0/",
                            author="Mihai Oltean", box=None, fixed_split=sp,
                            loader=lambda z=z, n=n: Image.open(io.BytesIO(z.read(n))).convert("RGB")))
    return out, sorted(set(c for c in cls if c not in keep))


def fetch(url):
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            pass
    return None


def main():
    info = {}
    ro, skip_lic, nbad = coco()
    info["coco_skipped_NoDerivs_instances"] = skip_lic
    info["coco_images_excluded_food_or_apple_banana"] = nbad
    rr, noi = oi()
    info["oi_images_excluded_apple_banana_tomato_food"] = noi
    rf, f360_excluded = f360()
    info["f360_classes_excluded"] = f360_excluded
    allr = ro + rr + rf
    for r in allr:
        r["split"] = r.get("fixed_split") or split_of(r["pid"])
        r["group"] = GROUP.get(r["category"], "household")
    # quotas (train, val, test); shortfalls are filled from the groups that have more
    quota = {"produce": 0.25, "produce_f360": 0.05, "round": 0.10, "electronics": 0.15, "household": 0.12,
             "shoe_toy": 0.08, "people": 0.08, "animal_car": 0.07, "scene": 0.05}
    size = {"train": 1500, "val": 300, "test": 300}
    chosen = []
    for sp, n in size.items():
        pool = {g: [r for r in allr if r["split"] == sp and r["group"] == g] for g in quota}
        for g in pool:
            rnd.shuffle(pool[g])
        # in a group spread over categories: round-robin
        take = {}
        for g, share in quota.items():
            bycat = {}
            for r in pool[g]:
                bycat.setdefault(r["category"], []).append(r)
            order, k = [], 0
            while any(bycat.values()):
                for c in sorted(bycat):
                    if bycat[c]:
                        order.append(bycat[c].pop())
            take[g] = order
        got = {g: take[g][: round(n * quota[g])] for g in quota}
        short = n - sum(len(v) for v in got.values())
        while short > 0:
            moved = False
            for g in quota:
                if short > 0 and len(got[g]) < len(take[g]):
                    got[g].append(take[g][len(got[g])])
                    short -= 1
                    moved = True
            if not moved:
                break
        for g in got:
            chosen += got[g]
    # no source photo in two splits (crops of one photo share the split by construction); check
    seen = {}
    for r in chosen:
        assert seen.setdefault(r["pid"], r["split"]) == r["split"], r["pid"]
    for sp in size:
        (OUT / sp).mkdir(parents=True, exist_ok=True)

    def work(r):
        im = r["loader"]() if "loader" in r else fetch(r["url"])
        if im is None:
            return None
        if r["box"] is not None:
            b = r["box"]
            if r.get("rel"):
                b = (b[0] * im.width, b[1] * im.height, b[2] * im.width, b[3] * im.height)
            im = crop_box(im, *b)
        im = shrink(im)
        p = OUT / r["split"] / (r["key"] + ".jpg")
        im.save(p, quality=92)
        return p

    # zip readers are not thread safe: COCO/F360 sequentially, downloads in threads
    done = []
    local = [r for r in chosen if "loader" in r]
    remote = [r for r in chosen if "loader" not in r]
    for r in local:
        done.append((r, work(r)))
    with ThreadPoolExecutor(16) as ex:
        for r, p in zip(remote, ex.map(work, remote)):
            done.append((r, p))
    failed = [r["key"] for r, p in done if p is None]
    info["download_failed"] = failed
    with open(ROOT / "manifest_r39.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "category", "group", "source", "source_id", "licence", "author", "box"])
        for r, p in sorted(done, key=lambda t: (t[0]["split"], t[0]["key"])):
            if p:
                w.writerow([r["split"], str(p), r["category"], r["group"], r["source"], r["source_id"], r["licence"],
                            r["author"], "" if r["box"] is None else json.dumps(r["box"])])
    json.dump(info, open(ROOT / "build_info_r39.json", "w"), indent=1)
    print(json.dumps(info, indent=1)[:2000])


main()

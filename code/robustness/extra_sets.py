"""Banana robustness evaluation sets (fixed seeds, written with manifests):
- bg_new2: banana photo shrunk to 50-80% on crops of 13 other real photos (not used for bg_new), appended to manifest_{split}.csv
- framing sets (separate manifests): small (photo shrunk to 20-40% of the frame), cut (banana shifted so about 30% of it leaves the frame),
  two (two photos of different stages side by side; label = the left photo's, both labels recorded)."""

import glob
import os

import bgaug
import numpy as np
import pandas as pd
from common import CLASSES, OUT_DIR, load_split
from hardset import HARD
from PIL import Image

MASKS = f"{OUT_DIR}/masks"


def real_bgs(folder):
    return [
        Image.open(f).convert("RGB")
        for f in sorted(glob.glob(f"{OUT_DIR}/{folder}/*.jpg"))
    ]


def crop_bg(rng, srcs, size):
    s = srcs[int(rng.integers(0, len(srcs)))]
    cs = int(rng.uniform(0.4, 1.0) * min(s.size))
    x0 = int(rng.integers(0, s.width - cs + 1))
    y0 = int(rng.integers(0, s.height - cs + 1))
    bg = s.crop((x0, y0, x0 + cs, y0 + cs)).resize(size, Image.BILINEAR)
    return bg.transpose(Image.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else bg


def build_bgnew2():
    srcs = real_bgs("bgnew2_src")
    for split, seed in (("val", 5321), ("test", 5322)):
        m = pd.read_csv(f"{HARD}/manifest_{split}.csv")
        m = m[m.kind != "bg_new2"]
        paths, y, df = load_split(split)
        rng = np.random.default_rng(seed)
        rows = []
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            iid = df.image_id[i].replace("/", "_").replace(".jpg", "")
            for c in range(2):
                bg = crop_bg(rng, srcs, img.size)
                sc = float(rng.uniform(0.5, 0.8))
                mm = int(img.width * sc)
                o = (img.width - mm) // 2
                bg.paste(img.resize((mm, mm), Image.BILINEAR), (o, o))
                f = f"{split}/{iid}_bgnew2{c}.png"
                bg.save(f"{HARD}/{f}")
                rows.append(
                    (f, df.image_id[i], CLASSES[y[i]], "bg_new2", f"scale={sc:.2f}")
                )
        pd.concat([m, pd.DataFrame(rows, columns=m.columns)]).to_csv(
            f"{HARD}/manifest_{split}.csv", index=False
        )
        print(split, "bg_new2", len(rows))


def shift_cut(img, mask, rng, frac=0.3):
    """Shift the photo (edge-replicated) so about `frac` of the banana mask leaves the frame."""
    a = np.asarray(img)
    side = int(rng.integers(0, 4))
    mm = mask if side < 2 else mask.T
    cum = np.cumsum(mm.sum(0)) / max(mm.sum(), 1)  # share of mask left of each column
    if side % 2 == 0:
        k = int(np.searchsorted(cum, frac))
        sh = a if side < 2 else a.transpose(1, 0, 2)
    else:
        k = int(np.searchsorted(1 - cum[::-1], frac))
        sh = a if side < 2 else a.transpose(1, 0, 2)
    pad = k + 1
    if side % 2 == 0:
        out = np.pad(sh, ((0, 0), (0, pad), (0, 0)), mode="edge")[:, pad:]
    else:
        out = np.pad(sh, ((0, 0), (pad, 0), (0, 0)), mode="edge")[:, :-pad]
    if side >= 2:
        out = out.transpose(1, 0, 2)
    return Image.fromarray(np.ascontiguousarray(out))


def build_framing():
    srcs = real_bgs("bgnew2_src")
    man = pd.read_csv(f"{MASKS}/manifest.csv").set_index("image_id")
    for split, seed in (("val", 7001), ("test", 7002)):
        paths, y, df = load_split(split)
        rng = np.random.default_rng(seed)
        rows = []
        os.makedirs(f"{HARD}/framing_{split}", exist_ok=True)
        imgs = [Image.open(p).convert("RGB") for p in paths]
        for i, img in enumerate(imgs):
            iid = df.image_id[i].replace("/", "_").replace(".jpg", "")
            lab = CLASSES[y[i]]
            # small: photo shrunk to 20-40% of the frame width, on a real-photo or procedural background
            s = float(rng.uniform(0.2, 0.4))
            m = int(256 * s)
            bg = (
                crop_bg(rng, srcs, (256, 256))
                if i % 2 == 0
                else bgaug.make_bg(
                    rng, 256, str(rng.choice(["solid", "stripes", "noise"]))
                )
            )
            bg.paste(
                img.resize((m, m), Image.BILINEAR),
                (int(rng.integers(0, 256 - m + 1)), int(rng.integers(0, 256 - m + 1))),
            )
            f = f"framing_{split}/{iid}_small.png"
            bg.save(f"{HARD}/{f}")
            rows.append((f, df.image_id[i], lab, "small", "", f"scale={s:.2f}"))
            # cut: banana partly outside the frame
            mk = (
                np.asarray(
                    Image.open(f"{MASKS}/{man.file[df.image_id[i]]}").resize(
                        (256, 256), Image.NEAREST
                    )
                )
                > 127
            )
            f = f"framing_{split}/{iid}_cut.png"
            shift_cut(img.resize((256, 256)), mk, rng).save(f"{HARD}/{f}")
            rows.append((f, df.image_id[i], lab, "cut", "", "shift 30% off frame"))
            # two bananas of different stages: left photo i, right photo j
            j = int(rng.choice([k for k in range(len(imgs)) if y[k] != y[i]]))
            cv = Image.new("RGB", (256, 256), (236, 236, 236))
            cv.paste(img.resize((128, 128)), (0, 64))
            cv.paste(imgs[j].resize((128, 128)), (128, 64))
            f = f"framing_{split}/{iid}_two.png"
            cv.save(f"{HARD}/{f}")
            rows.append(
                (
                    f,
                    df.image_id[i],
                    lab,
                    "two",
                    CLASSES[y[j]],
                    f"left {lab}, right {CLASSES[y[j]]}",
                )
            )
        pd.DataFrame(
            rows, columns=["file", "image_id", "label", "kind", "label_right", "params"]
        ).to_csv(f"{HARD}/manifest_framing_{split}.csv", index=False)
        print(split, "framing", len(rows))


def build_small_val2():
    """Validation small-frame set with the same recipe and a new seed (8001)."""
    srcs = real_bgs("bgnew2_src")
    paths, y, df = load_split("val")
    rng = np.random.default_rng(8001)
    rows = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        iid = df.image_id[i].replace("/", "_").replace(".jpg", "")
        s = float(rng.uniform(0.2, 0.4))
        m = int(256 * s)
        bg = (
            crop_bg(rng, srcs, (256, 256))
            if i % 2 == 0
            else bgaug.make_bg(rng, 256, str(rng.choice(["solid", "stripes", "noise"])))
        )
        bg.paste(
            img.resize((m, m), Image.BILINEAR),
            (int(rng.integers(0, 256 - m + 1)), int(rng.integers(0, 256 - m + 1))),
        )
        f = f"framing_val/{iid}_small2.png"
        bg.save(f"{HARD}/{f}")
        rows.append((f, df.image_id[i], CLASSES[y[i]], "small", "", f"scale={s:.2f}"))
    pd.DataFrame(
        rows, columns=["file", "image_id", "label", "kind", "label_right", "params"]
    ).to_csv(f"{HARD}/manifest_small2_val.csv", index=False)
    print("small2 val", len(rows))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "small2":
        build_small_val2()
    else:
        build_bgnew2()
        build_framing()

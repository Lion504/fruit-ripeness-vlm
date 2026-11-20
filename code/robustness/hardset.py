"""Build the FIXED hard evaluation sets (seed 1234) for val and test, written with a manifest.
Kinds: clean (not stored), bg (2 copies/image: shrunk 50-80% on varied backgrounds), wb (warm + cool), blur (blur + JPEG q50)."""

import os
import sys

import bgaug
import numpy as np
import pandas as pd
from common import CLASSES, OUT_DIR, load_split
from PIL import Image

HARD = f"{OUT_DIR}/hard"
NAMES = list(bgaug.NAMED) + ["stripes", "noise"]


def build():
    for split, seed in (("val", 1234), ("test", 1235)):
        paths, y, df = load_split(split)
        rng = np.random.default_rng(seed)
        rows = []
        os.makedirs(f"{HARD}/{split}", exist_ok=True)
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            iid = df.image_id[i].replace("/", "_").replace(".jpg", "")
            outs = []
            for c in range(2):
                name = NAMES[(2 * i + c + int(rng.integers(0, 7))) % 7]
                s = float(rng.uniform(0.5, 0.8))
                if name in bgaug.NAMED:
                    im = bgaug.paste_on_bg(img, rng, "solid", bgaug.NAMED[name], s)
                else:
                    im = bgaug.paste_on_bg(img, rng, name, None, s)
                outs.append((f"bg{c}", im, f"{name} scale={s:.2f}"))
            outs.append(("wb0", bgaug.white_balance(img, True), "warm"))
            outs.append(("wb1", bgaug.white_balance(img, False), "cool"))
            outs.append(("blur", bgaug.blur_jpeg(img), "gauss1.2+jpegq50"))
            for kind, im, par in outs:
                f = (
                    f"{split}/{iid}_{kind}.png"
                    if kind != "blur"
                    else f"{split}/{iid}_{kind}.jpg"
                )
                if kind == "blur":
                    im.save(f"{HARD}/{f}", quality=95)
                else:
                    im.save(f"{HARD}/{f}")
                rows.append(
                    (
                        f,
                        df.image_id[i],
                        CLASSES[y[i]],
                        kind[:2] if kind.startswith(("bg", "wb")) else kind,
                        par,
                    )
                )
        pd.DataFrame(
            rows, columns=["file", "image_id", "label", "kind", "params"]
        ).to_csv(f"{HARD}/manifest_{split}.csv", index=False)
        print(split, len(rows))


def load_hard(split):
    """Return {kind: (images, labels)} for kinds bg, wb, blur."""
    m = pd.read_csv(f"{HARD}/manifest_{split}.csv")
    out = {}
    for k, g in m.groupby("kind"):
        out[k] = (
            [Image.open(f"{HARD}/{f}").convert("RGB") for f in g.file],
            g.label.map(CLASSES.index).to_numpy(),
        )
    return out


def build_bgnew():
    """Add the bg_new kind: banana photo shrunk to 50-80% and pasted on random crops of real, non-banana photos
    (a few photos without bananas, in $OUT/bgnew_src; not redistributable, so not published). Fixed seeds; appended to the manifests."""
    import glob

    from sklearn.datasets import load_sample_images

    srcs = [Image.fromarray(a) for a in load_sample_images().images]
    srcs += [
        Image.open(f).convert("RGB")
        for f in sorted(glob.glob(f"{OUT_DIR}/bgnew_src/*.jpg"))
    ]
    for split, seed in (("val", 4321), ("test", 4322)):
        m = pd.read_csv(f"{HARD}/manifest_{split}.csv")
        m = m[m.kind != "bg_new"]
        paths, y, df = load_split(split)
        rng = np.random.default_rng(seed)
        rows = []
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            iid = df.image_id[i].replace("/", "_").replace(".jpg", "")
            for c in range(2):
                si = int(rng.integers(0, len(srcs)))
                s = srcs[si]
                cs = int(rng.uniform(0.4, 1.0) * min(s.size))
                x0 = int(rng.integers(0, s.width - cs + 1))
                y0 = int(rng.integers(0, s.height - cs + 1))
                bg = s.crop((x0, y0, x0 + cs, y0 + cs)).resize(img.size, Image.BILINEAR)
                if rng.random() < 0.5:
                    bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
                sc = float(rng.uniform(0.5, 0.8))
                mm = int(img.width * sc)
                o = (img.width - mm) // 2
                bg.paste(img.resize((mm, mm), Image.BILINEAR), (o, o))
                f = f"{split}/{iid}_bgnew{c}.png"
                bg.save(f"{HARD}/{f}")
                rows.append(
                    (
                        f,
                        df.image_id[i],
                        CLASSES[y[i]],
                        "bg_new",
                        f"src{si} crop{cs} scale={sc:.2f}",
                    )
                )
        pd.concat([m, pd.DataFrame(rows, columns=m.columns)]).to_csv(
            f"{HARD}/manifest_{split}.csv", index=False
        )
        print(split, "bg_new", len(rows), "sources", len(srcs))


def load_hard_ids(split):
    """image_id per image for each kind, in the same order as load_hard."""
    m = pd.read_csv(f"{HARD}/manifest_{split}.csv")
    return {k: g.image_id.tolist() for k, g in m.groupby("kind")}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bgnew":
        build_bgnew()
    else:
        build()

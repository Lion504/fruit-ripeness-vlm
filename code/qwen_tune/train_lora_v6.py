"""Shared LoRA stage trainer (v6): one adapter per fruit on the language layers, vision frozen.

usage: train_lora_v6.py FRUIT            FRUIT = banana | apple | tomato | pass1 (pass1: classes banana/apple/tomato/none/unclear)
env:   RIPENESS_TRAINING_DIR, RIPENESS_BANANA_DATASET, RIPENESS_QWEN_BASE, HF_HOME (paths, see paths.py); device = cuda if available else mps (bf16 either way)
       EPOCHS=8  LR=1e-4  RANK=16  ACC=8 (samples per optimizer step)  WARM=40 (optimizer steps)  LOG_EVERY=10 (optimizer steps)
       LOWRES=1 (round 31: small-in-frame pasting and low-resolution round trips in the augmentation)
       EXTRA_TRAIN=<csv> (round 32: extra train crops, columns path,label, added to the fruit's own train rows;
                          for pass1 the label is the fruit name)
       PARTIAL=1 (round 33: 15% of random crops keep only 40-75% of a side, so cut-off fruit is seen in training)
       NEG_EXTRA=<csv> (round 39, pass1 only: extra "none" photos, columns path (any other columns ignored); every row is
                        added once per epoch with label none and the normal augmentation. Unset = rounds 29-38 behaviour)
       OUT=<dir> (default qwen-tune/lora_v6/<fruit>)   SEED=0   MAX_SAMPLES=N (stop early; capacity tests)
Recipe: all train crops (apple/tomato: truncated and small crops kept), augmentation (flip, small rotation with
background-pool fill, random crop, brightness/contrast/saturation jitter, mild blur, 'unclear' samples made from
darkened or heavily blurred fruit), class weights (inverse frequency, capped), lr warm-up then cosine, loss on the
stage value tokens only. Checkpoint (PEFT adapter) every epoch: OUT/ep<N>/. At each epoch end the (non-augmented)
train accuracy on a fixed 200-crop subset is computed and the per-crop predictions are appended to OUT/train_subset_preds.jsonl.
Vision tower and merger are frozen; LoRA targets only language-layer attention (6 full-attention layers) and MLP
(all 24 layers): the linear-attention projections cannot be converted to a GGUF adapter by the converter.
"""

import csv
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lora_common_v6 import TARGET_REGEX
from paths import BASE_MODEL, CROPS, QWEN_DIR, SPLITS_CSV, banana_path
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageEnhance, ImageFilter
from produce_common import TOMATO_MAP, prep_image
from v6_common import PASS1_ENUM, PASS1_PROMPT, PASS1_USER, STAGE_ENUM, stage_prompt

STAGE_ENUM = dict(
    STAGE_ENUM, pass1=PASS1_ENUM
)  # 'pass1' = the fruit-identification adapter

fruit = sys.argv[1]
assert fruit in STAGE_ENUM
env = os.environ.get
EPOCHS, LR, RANK, ACC = (
    int(env("EPOCHS", "8")),
    float(env("LR", "1e-4")),
    int(env("RANK", "16")),
    int(env("ACC", "8")),
)
WARM, LOG_EVERY, SEED = (
    int(env("WARM", "40")),
    int(env("LOG_EVERY", "10")),
    int(env("SEED", "0")),
)
MAX_SAMPLES = int(env("MAX_SAMPLES", "0"))
LOWRES = env("LOWRES", "0") == "1"  # round 31 small-fruit and low-resolution augmentation
EXTRA_TRAIN = env("EXTRA_TRAIN", "")  # round 32: CSV (path,label) of extra train crops
PARTIAL = env("PARTIAL", "0") == "1"  # round 33: 15% of random crops keep only 40-75% of a side
NEG_EXTRA = env("NEG_EXTRA", "")  # round 39: CSV (path) of extra non-fruit photos, label none (pass1 only)
QT = QWEN_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
out = Path(env("OUT", str(QT / "lora_v6" / fruit)))
out.mkdir(parents=True, exist_ok=True)


def read_rows(produce, split, drop_truncated=True, min_side=48):
    res = []
    for r in csv.DictReader(open(CROPS / produce / f"{split}.csv")):
        if drop_truncated and (
            r["truncated"] == "1" or min(int(r["width"]), int(r["height"])) < min_side
        ):
            continue
        res.append(
            (
                str(CROPS / produce / r["crop_path"]),
                TOMATO_MAP[r["class"]] if produce == "tomato" else r["class"],
                r["class"],
            )
        )
    return res


def load_split(split):
    res = []
    for r in csv.DictReader(open(SPLITS_CSV)):
        if r["split"] == split:
            res.append((banana_path(r["path"]), r["label"]))
    return res


def bgpool(produce, lo, hi):
    return [str(CROPS / f"{produce}_bgpool" / f"bg_{i:04d}.jpg") for i in range(lo, hi)]


def load_model(dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = str(BASE_MODEL)
    return AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).to(
        DEVICE
    ), AutoProcessor.from_pretrained(base)


STAGES = STAGE_ENUM[fruit][:-1]  # real stages; 'unclear' is built from degraded images
SYSTEM = PASS1_PROMPT if fruit == "pass1" else stage_prompt(fruit)
USER = PASS1_USER if fruit == "pass1" else "What is the ripeness stage?"
KEY = "produce" if fruit == "pass1" else "stage"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_rows():
    if fruit == "pass1":
        rr = random.Random(11)
        ban = [(p, "banana") for p, _ in load_split("train")]
        ap = [
            (p, "apple")
            for p, _, _ in read_rows("apple", "train", drop_truncated=False)
        ]
        to = [
            (p, "tomato")
            for p, _, _ in read_rows("tomato", "train", drop_truncated=False)
        ]
        rows = rr.sample(ban, min(1500, len(ban))) + ap + to
        if EXTRA_TRAIN:  # round 33: extra fruit photos (CSV path,label with label = fruit name)
            rows += [(r["path"], r["label"]) for r in csv.DictReader(open(EXTRA_TRAIN))]
        # none: train-range background patches (indices 0-119) x5 augmented draws + procedural plain/noise images
        rows += [("bg:" + p, "none") for p in BGTRAIN for _ in range(5)] + [
            ("proc:%d" % rr.randrange(10**6), "none") for _ in range(150)
        ]
        # unclear: darkened / heavily blurred fruit
        rows += [("dark:" + p, "unclear") for p, _ in rr.sample(rows[:3000], 400)]
        if NEG_EXTRA:  # round 39: everyday objects, other produce, look-alikes (added after the unclear draw, so that draw is unchanged)
            rows += [(r["path"], "none") for r in csv.DictReader(open(NEG_EXTRA))]
        return rows
    if fruit == "banana":
        rows = [(p, lbl) for p, lbl in load_split("train")]
    else:
        rows = [(p, lab) for p, lab, _ in read_rows(fruit, "train", drop_truncated=False)]
    if EXTRA_TRAIN:  # round 32: extra train crops from another dataset (CSV with path,label)
        rows += [(r["path"], r["label"]) for r in csv.DictReader(open(EXTRA_TRAIN))]
    return rows


BGTRAIN = bgpool("apple", 0, 120) + bgpool("tomato", 0, 120)
rows = load_rows()
if env("ROWS_LIMIT"):  # smoke tests only
    rows = random.Random(1).sample(rows, int(env("ROWS_LIMIT")))
cnt = {
    s: sum(1 for _, lbl in rows if lbl == s)
    for s in (PASS1_ENUM if fruit == "pass1" else STAGES)
}
STAGES = list(cnt)
# class weights: inverse frequency normalised to mean 1 over samples, capped at 3
W = {s: min(3.0, len(rows) / (len(STAGES) * cnt[s])) for s in STAGES}
W["unclear"] = 1.0
BG = bgpool("apple", 0, 120) + bgpool("tomato", 0, 120)  # train range of the pools only
sub_rng = random.Random(0)
SUBSET = sub_rng.sample(rows, min(int(env("SUBSET_N", "200")), len(rows)))
print(
    "fruit",
    fruit,
    "train rows",
    len(rows),
    "counts",
    cnt,
    "weights",
    {k: round(v, 2) for k, v in W.items()},
    flush=True,
)


def augment(img, rng):
    w, h = img.size
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # random crop (keeps 75-100% of each side)
    lo = 0.4 if PARTIAL and rng.random() < 0.15 else 0.75  # round 33: sometimes only part of the fruit
    cw, ch = int(w * rng.uniform(lo, 1)), int(h * rng.uniform(lo, 1))
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    # small rotation; the corners are filled from a background-pool patch (or a plain colour)
    if rng.random() < 0.6:
        ang = rng.uniform(-20, 20)
        rgba = img.convert("RGBA").rotate(ang, expand=True, resample=Image.BICUBIC)
        if rng.random() < 0.6:
            bg = Image.open(rng.choice(BG)).convert("RGB").resize(rgba.size)
        else:
            bg = Image.new(
                "RGB", rgba.size, tuple(rng.randrange(0, 255) for _ in range(3))
            )
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    # background pasting: shrink the fruit crop onto a pool patch
    if rng.random() < 0.2:
        bg = (
            Image.open(rng.choice(BG))
            .convert("RGB")
            .resize((int(img.width / 0.8), int(img.height / 0.8)))
        )
        bg.paste(img, ((bg.width - img.width) // 2, (bg.height - img.height) // 2))
        img = bg
    if LOWRES:
        # round 31: small fruit in a wider frame, and low-resolution photos (e.g. 160 px crops) scaled back up,
        # so that small or soft photos keep their stage instead of reading as 'unclear'
        if rng.random() < 0.15:
            s = rng.uniform(0.35, 0.6)
            cap = 768 / (max(img.size) / s)  # keep the pasted photo's long side at 768 px or less
            if cap < 1:
                img = img.resize((max(1, int(img.width * cap)), max(1, int(img.height * cap))), Image.BILINEAR)
            bg = Image.open(rng.choice(BG)).convert("RGB").resize((int(img.width / s), int(img.height / s)))
            bg.paste(img, (rng.randint(0, bg.width - img.width), rng.randint(0, bg.height - img.height)))
            img = bg
        if rng.random() < 0.3:
            short = rng.randint(48, 160)
            f = short / min(img.size)
            if f < 1:
                small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
                img = small.resize(img.size, Image.BICUBIC)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(
        rng.uniform(0.85, 1.15)
    )  # no hue shift: hue carries the ripeness
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    return img


def degrade(img, rng):
    """'unclear' sample: very dark or heavily blurred fruit."""
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=10 + rng.randrange(5)))
        return ImageEnhance.Brightness(img).enhance(rng.choice([0.005, 0.02, 0.05]))
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(14, 24)))


def proc_bg(seed, size=256):
    r = random.Random(seed)
    col = np.array([r.randrange(30, 250) for _ in range(3)], np.float32)
    a = np.ones((size, size, 3), np.float32) * col
    if r.random() < 0.4:
        a *= (1 + 0.2 * np.sin(np.arange(size) / r.randrange(6, 40) * 6.28))[
            None, :, None
        ]
    elif r.random() < 0.7:
        a += np.random.default_rng(seed).normal(0, 20, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_sample(path, label, rng, aug=True):
    if path.startswith("proc:"):
        return prep_image(proc_bg(int(path[5:]))), label
    if path.startswith("dark:"):
        r2 = rng or random.Random(0)
        return prep_image(degrade(Image.open(path[5:]).convert("RGB"), r2)), label
    if path.startswith("bg:"):
        img = Image.open(path[3:]).convert("RGB")
        return prep_image(augment(img, rng) if aug else img), label
    img = Image.open(path).convert("RGB")
    if aug:
        if fruit != "pass1" and rng.random() < 0.04:  # unclear
            return prep_image(degrade(img, rng)), "unclear"
        img = augment(img, rng)
    return prep_image(img), label


model, proc = load_model(dtype=torch.bfloat16)
cfg = LoraConfig(
    r=RANK, lora_alpha=2 * RANK, target_modules=TARGET_REGEX, lora_dropout=0.05
)
model = get_peft_model(model, cfg)
for n, p in model.named_parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
params = [p for p in model.parameters() if p.requires_grad]
print("trainable", sum(p.numel() for p in params), flush=True)
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
tok = proc.tokenizer
OPTIONS = STAGE_ENUM[fruit]
first = {o: tok.encode(o, add_special_tokens=False)[0] for o in OPTIONS}
assert len(set(first.values())) == len(OPTIONS), f"first tokens not distinct: {first}"
PREFIX = '{"%s": "' % KEY
steps_total = EPOCHS * math.ceil(len(rows) / ACC)


def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    return (
        LR * 0.5 * (1 + math.cos(math.pi * (step - WARM) / max(1, steps_total - WARM)))
    )


def inputs(img, answer=""):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [{"type": "text", "text": USER}, {"type": "image"}],
        },
    ]
    text = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return proc(text=[text + PREFIX + answer], images=[img], return_tensors="pt").to(
        DEVICE
    )


def sample_loss(img, label):
    ans = f'{label}"}}'
    inp = inputs(img, ans)
    n = len(tok.encode(ans, add_special_tokens=False))
    logits = model(**inp).logits[0, -n - 1 : -1].float()
    tgt = inp["input_ids"][0, -n:]
    ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
    return ce.mean() * W[label]


@torch.no_grad()
def predict(img):
    lg = model(**inputs(img)).logits[0, -1].float()
    sc = {o: lg[first[o]].item() for o in OPTIONS}
    return max(sc, key=sc.get), sc


def mem():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1e6 if sys.platform == "linux" else 1e9
    )  # KB on Linux, bytes on macOS
    if DEVICE == "cuda":
        return rss, torch.cuda.max_memory_allocated() / 1e9
    return rss, torch.mps.driver_allocated_memory() / 1e9


def subset_eval(ep):
    model.eval()
    ok = 0
    with open(out / "train_subset_preds.jsonl", "a") as f:
        for p, lbl in SUBSET:
            img, _ = make_sample(p, lbl, None, aug=False)
            pred, sc = predict(img)
            ok += pred == lbl
            f.write(
                json.dumps(
                    {"epoch": ep, "path": p, "label": lbl, "pred": pred, "scores": sc}
                )
                + "\n"
            )
    model.train()
    return ok / len(SUBSET)


model.train()
seen = step = 0
run = 0.0
rc = 0
t0 = time.time()
tstep = time.time()
peak_rss = peak_mps = 0.0
json.dump(
    {
        "fruit": fruit,
        "rows": len(rows),
        "counts": cnt,
        "weights": W,
        "epochs": EPOCHS,
        "lr": LR,
        "rank": RANK,
        "acc": ACC,
        "warm": WARM,
        "init": "base",
        "lowres": LOWRES,
        "extra_train": EXTRA_TRAIN,
        "partial": PARTIAL,
        "neg_extra": NEG_EXTRA,
    },
    open(out / "config.json", "w"),
)
for ep in range(1, EPOCHS + 1):
    rng = random.Random(SEED * 1000 + ep)
    order = list(range(len(rows)))
    rng.shuffle(order)
    for k, i in enumerate(order):
        p, lbl = rows[i]
        img, lab = make_sample(p, lbl, rng)
        loss = sample_loss(img, lab)
        (loss / ACC).backward()
        run += loss.item()
        rc += 1
        seen += 1
        if (k + 1) % ACC == 0 or k == len(order) - 1:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if DEVICE == "mps":
                torch.mps.empty_cache()
            r, m = mem()
            peak_rss, peak_mps = max(peak_rss, r), max(peak_mps, m)
            if step % LOG_EVERY == 0:
                dt = time.time() - tstep
                tstep = time.time()
                print(
                    f"ep {ep} step {step}/{steps_total} samples {seen} loss {run / rc:.4f} lr {lr_at(step):.2e} "
                    f"s/step {dt / LOG_EVERY:.2f} s/sample {dt / LOG_EVERY / ACC:.3f} peak_rss {peak_rss:.2f}GB mps {peak_mps:.2f}GB",
                    flush=True,
                )
                run = 0.0
                rc = 0
        if MAX_SAMPLES and seen >= MAX_SAMPLES:
            print(
                f"STOP at MAX_SAMPLES; total {time.time() - t0:.0f}s peak_rss {peak_rss:.2f}GB peak_mps {peak_mps:.2f}GB",
                flush=True,
            )
            sys.exit(0)
    model.save_pretrained(out / f"ep{ep}")
    acc = subset_eval(ep)
    print(
        f"EPOCH {ep} done t={time.time() - t0:.0f}s train-subset(200) acc {acc:.3f} -> {out}/ep{ep}",
        flush=True,
    )
print("TRAIN DONE", flush=True)

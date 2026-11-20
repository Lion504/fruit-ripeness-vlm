"""Run one public model over one test set and save raw predictions.

usage: predict.py MODEL TESTSET OUT.jsonl
MODEL   = carlos | dhahlan | parsa | dima806 | aryap2 | reitk | tcleo | deanza | afeef
TESTSET = banana | apple | tomato | parsa_tomato_test | frame_small | frame_cut | bg_new2 | parsa_test
  (the last four read TEST_LISTS_DIR/<name>.jsonl; parsa_test = their 721 non-rotten test crops, label already
  not_yet/ready)
Each output row: path, label (our label, or their class for parsa_tomato_test), source, raw (top class name),
conf (top probability), probs (class -> probability).
afeef is a detector: raw = class of the highest-confidence box (conf >= 0.25, Ultralytics default) or 'unclear'
if there is none; probs = best box confidence per class (boxes down to conf 0.05), n_boxes = boxes >= 0.25.
"""

import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# dataset paths come from RIPEWISE_TRAINING_DIR, BANANA_SPLITS and RIPEWISE_BANANA_DATASET (see code/qwen_tune/paths.py)
sys.path.insert(0, str(REPO / "code" / "qwen_tune"))
from paths import CROPS, SPLITS_CSV, banana_path  # noqa: E402
from produce_common import TOMATO_MAP  # noqa: E402

MODELS = Path(os.environ.get("COMPARE_MODELS_DIR", REPO / "compare_models"))  # downloaded public models
PARSA_TEST = REPO / "data" / "parsavictor" / "cls_tomato" / "test"  # or set PARSA_TEST
LISTS = Path(os.environ.get("TEST_LISTS_DIR", REPO / "data" / "lists"))  # frame_small, frame_cut, bg_new2, parsa_test .jsonl


def rows_for(testset):
    if testset == "banana":
        return [(banana_path(r["path"]), r["label"], r["source"])
                for r in csv.DictReader(open(SPLITS_CSV)) if r["split"] == "test"]
    if testset in ("apple", "tomato"):
        out = []
        for r in csv.DictReader(open(CROPS / testset / "test.csv")):
            lab = TOMATO_MAP[r["class"]] if testset == "tomato" else r["class"]
            out.append((str(CROPS / testset / r["crop_path"]), lab, testset))
        return out
    if testset == "parsa_tomato_test":
        root = Path(os.environ.get("PARSA_TEST", PARSA_TEST))
        return [(str(p), p.parent.name, "parsavictor_test")
                for p in sorted(root.glob("*/*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    if testset in ("frame_small", "frame_cut", "bg_new2", "parsa_test"):
        rows = [json.loads(line) for line in open(LISTS / f"{testset}.jsonl")]
        return [(r["path"], r["label"], r["source"]) for r in rows]
    raise SystemExit(testset)


def carlos():
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from PIL import Image
    d = MODELS / "Carlosmg23__fruit-ripeness-final"
    classes = json.load(open(d / "clases.json"))["clases"]
    it = Interpreter(model_path=str(d / "modelo_final.tflite"))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    norm = os.environ.get("CARLOS_NORM", "raw")  # raw = 0..255 floats (graph has its own Rescaling); mobilenet = x/127.5-1

    def f(path):
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32)
        if norm == "mobilenet":
            x = x / 127.5 - 1.0
        it.set_tensor(inp["index"], x[None])
        it.invoke()
        p = it.get_tensor(out["index"])[0].astype(float)
        if p.min() < 0 or abs(p.sum() - 1) > 1e-3:  # logits -> softmax
            p = np.exp(p - p.max())
            p /= p.sum()
        return dict(zip(classes, p.tolist()))
    return f


def hf_vit(name):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    d = MODELS / name
    try:
        proc = AutoImageProcessor.from_pretrained(d)
    except ValueError:  # old 'ViTFeatureExtractor' type name (Dhahlan2000); same settings via ViTImageProcessor
        from transformers import ViTImageProcessor
        proc = ViTImageProcessor.from_pretrained(d)
    m = AutoModelForImageClassification.from_pretrained(d).eval().to("cuda")
    lab = m.config.id2label

    @torch.no_grad()
    def f(path):
        x = proc(images=Image.open(path).convert("RGB"), return_tensors="pt").to("cuda")
        p = m(**x).logits.softmax(-1)[0].tolist()
        return {lab[i]: v for i, v in enumerate(p)}
    return f


def parsa():
    from ultralytics import YOLO
    m = YOLO(str(MODELS / "ParsaVictor__tomato-ripeness-classifier-yolo26s/ripeness_yolo26s-cls_best.pt"))
    imgsz = int(os.environ.get("PARSA_IMGSZ", "0")) or m.overrides.get("imgsz", 224)

    def f(path):
        r = m.predict(path, imgsz=imgsz, verbose=False, device=0)[0]
        return {r.names[i]: float(v) for i, v in enumerate(r.probs.data.tolist())}
    return f


def reitk():
    """Reitk/banana-ripeness-classifier (Space): torchvision resnet18 state_dict in a plain dict checkpoint, loaded
    with weights_only=True; preprocessing as in its src/banana_ripeness/model.py inference_transform
    (Resize((224, 224)), ToTensor, ImageNet mean/std). Their code is read, not imported."""
    import torch
    from PIL import Image
    from torchvision import models, transforms
    ck = torch.load(MODELS / "Reitk__banana-ripeness-classifier/models/banana_ripeness_resnet18.pt",
                    map_location="cpu", weights_only=True)
    classes = list(ck["class_names"])
    size = int(ck.get("image_size", 224))
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ck["model_state_dict"])
    m = m.eval().to("cuda")
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    @torch.no_grad()
    def f(path):
        x = tf(Image.open(path).convert("RGB"))[None].to("cuda")
        return dict(zip(classes, m(x).softmax(-1)[0].tolist()))
    return f


def afeef():
    from ultralytics import YOLO
    m = YOLO(str(MODELS / "afeefaazam__tomato-ripeness-detector/model.pt"))
    imgsz = int(os.environ.get("AFEEF_IMGSZ", "0")) or m.overrides.get("imgsz", 640)

    def f(path):
        r = m.predict(path, imgsz=imgsz, conf=0.05, verbose=False, device=0)[0]
        best, n = {}, 0
        for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = r.names[int(c)]
            best[name] = max(best.get(name, 0.0), float(s))
            n += s >= 0.25
        if not best or max(best.values()) < 0.25:
            best["unclear"] = 1.0  # no box at the default threshold
        return best, n
    return f


if __name__ == "__main__":
    model, testset, out = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # sanity runs: every k-th photo, about `limit` rows
    fn = {"carlos": carlos, "parsa": parsa, "reitk": reitk, "afeef": afeef,
          "dhahlan": lambda: hf_vit("Dhahlan2000__banana_ripeness_level_detection"),
          "dima806": lambda: hf_vit("dima806__fruit_vegetable_image_detection"),
          "aryap2": lambda: hf_vit("aryap2__kematangan-pisang-vit-b-32-100eph-224-v2.4"),
          "tcleo": lambda: hf_vit("TCleo__banana-maturity-mobile-vit-small"),
          "deanza": lambda: hf_vit("deanza1000__tomato-ripeness-resnet")}[model]()
    rows = rows_for(testset)
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fo:
        for i, (p, lab, src) in enumerate(rows):
            extra = {}
            probs = fn(p)
            if model == "afeef":
                probs, nb = probs
                extra["n_boxes"] = int(nb)
                top = "unclear" if "unclear" in probs else max(probs, key=probs.get)
            else:
                top = max(probs, key=probs.get)
            if model == "dima806":  # keep the file small: top 5 only
                probs = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:5])
            fo.write(json.dumps(dict(path=p, label=lab, source=src, raw=top, conf=round(probs[top], 5),
                                     probs={k: round(v, 5) for k, v in probs.items()}, **extra)) + "\n")
    print(model, testset, len(rows), "rows ->", out)

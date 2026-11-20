"""Scorer C: phone-form stage scorer for one fruit adapter (apple / tomato / banana).
usage: score_stage_v6.py FRUIT SPLIT OUT_DIR ENDPOINT [LORA_ID|none] [LIMIT]
SPLIT = val | test (apple/tomato from crops/<fruit>/<split>.csv, ALL crops incl. truncated; banana from splits.csv).
One request per crop: stage-only system prompt, strict schema {"stage": enum}, thinking off, temperature 0, seed 42;
the adapter is selected per request via "lora": [{"id": ID, "scale": 1}] (server started with --lora-init-without-apply
and the global scale set to 0). Writes OUT_DIR/results.jsonl (one row per crop: path, label, pred, ms, raw) and
OUT_DIR/score.json (computed from results.jsonl by metrics_from_results()). Resumable."""

import csv
import json
import statistics
import sys
from pathlib import Path

from paths import CROPS, SPLITS_CSV, banana_path
from produce_common import STAGES as ALL_STAGES
from produce_common import TOMATO_MAP
from v6_common import ask, b64_of, stage_format, stage_prompt


def rows_for(fruit, split):
    if fruit == "banana":
        return [
            (banana_path(r["path"]), r["label"], r.get("source", "bananaimagebd"))
            for r in csv.DictReader(open(SPLITS_CSV))
            if r["split"] == split
        ]
    out = []
    for r in csv.DictReader(open(CROPS / fruit / f"{split}.csv")):
        lab = TOMATO_MAP[r["class"]] if fruit == "tomato" else r["class"]
        out.append((str(CROPS / fruit / r["crop_path"]), lab, fruit))
    return out


def metrics_from_results(rows, fruit):
    st = ALL_STAGES[fruit]
    n = len(rows)
    conf = {a: {b: 0 for b in st + ["unclear", "invalid"]} for a in st}
    for r in rows:
        conf[r["label"]][r["pred"] if r["pred"] in st + ["unclear"] else "invalid"] += 1
    correct = sum(conf[c][c] for c in st)
    f1 = []
    for c in st:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in st) - tp
        fn = sum(conf[c].values()) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    valid = [r for r in rows if r["pred"] in st]
    d = [abs(st.index(r["pred"]) - st.index(r["label"])) for r in valid]
    ms = sorted(r["ms"] for r in rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "macro_f1": sum(f1) / len(st),
        "per_stage_recall": {c: [conf[c][c], sum(conf[c].values())] for c in st},
        "ordinal_mae": sum(d) / len(d) if d else None,
        "ordinal_mse": sum(x * x for x in d) / len(d) if d else None,
        "ordinal_n": len(d),
        "unclear": sum(conf[c]["unclear"] for c in st),
        "invalid": sum(conf[c]["invalid"] for c in st),
        "confusion": conf,
        "median_ms": statistics.median(ms),
        "p95_ms": ms[max(0, int(0.95 * len(ms)) - 1)],
    }


if __name__ == "__main__":
    fruit, split, out, ep = sys.argv[1:5]
    lid = sys.argv[5] if len(sys.argv) > 5 else "none"
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = rows_for(fruit, split)
    if limit:
        data = data[:limit]
    sysp, fmt = stage_prompt(fruit), stage_format(fruit)
    lora = None if lid == "none" else [{"id": int(lid), "scale": 1.0}]
    done = {}
    if (out / "results.jsonl").exists():
        done = {
            json.loads(line)["path"]: json.loads(line)
            for line in open(out / "results.jsonl")
        }
    f = open(out / "results.jsonl", "a")
    rows = []
    for i, (p, lab, src) in enumerate(data):
        if p in done:
            rows.append(done[p])
            continue
        raw = pred = None
        try:
            raw, ms, _ = ask(
                ep,
                sysp,
                "What is the ripeness stage?",
                fmt,
                b64_of(p),
                lora=lora,
                max_tokens=40,
                temperature=0.0,
            )
            pred = json.loads(raw)["stage"]
        except Exception as e:
            ms = 0
            raw = repr(e)[:100]
        d = {"path": p, "label": lab, "pred": pred, "ms": ms, "raw": raw, "source": src}
        f.write(json.dumps(d) + "\n")
        f.flush()
        rows.append(d)
        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(data), flush=True)
    res = metrics_from_results(rows, fruit)
    res.update(fruit=fruit, split=split, lora=lid)
    json.dump(res, open(out / "score.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}))

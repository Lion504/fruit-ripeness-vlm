"""Full metrics for one set of stage answers, and the round record.

Metrics (stages are ordered, e.g. unripe < turning < ripe < overripe):
- accuracy with a 95% Wilson interval; balanced accuracy (mean recall)
- per-stage precision, recall, F1 and support; macro F1 and weighted F1, with a 95% bootstrap interval for macro F1
- MAE and MSE in stages (how far off an answer is; answers outside the stage list are left out and counted)
- quadratic weighted Cohen's kappa (agreement that counts far-off answers more)
- confusion matrix; 'unclear' and invalid answers count as wrong for accuracy and recall

usage (record a scored run):
  full_metrics.py RECORD_CSV ROUND MODEL FRUIT SPLIT RESULTS.jsonl [NOTE]
  appends one row per data source plus an 'all' row to RECORD_CSV and writes RESULTS_DIR/full_metrics.json.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

STAGES = {
    "banana": ["unripe", "turning", "ripe", "overripe"],
    "apple": ["immature", "semi-mature", "mature"],
    "tomato": ["not_yet", "ready"],
    # merged label sets for comparing with public models that have fewer stages
    "banana3": ["unripe", "ripe", "overripe"],  # the 'turning' label merged into unripe or ripe (say which in the note)
    "apple2": ["immature", "mature"],  # semi-mature photos left out
}
FIELDS = [
    "round", "model", "fruit", "split", "source", "n", "correct", "accuracy", "acc_ci_low", "acc_ci_high",
    "balanced_accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "weighted_f1", "mae", "mse",
    "qw_kappa", "unclear", "invalid", "per_stage", "confusion", "note",
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _f1s(labels, preds, st):
    out = {}
    for c in st:
        tp = sum(1 for a, b in zip(labels, preds) if a == c and b == c)
        fp = sum(1 for a, b in zip(labels, preds) if a != c and b == c)
        fn = sum(1 for a, b in zip(labels, preds) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, support=tp + fn)
    return out


def qw_kappa(labels, preds, st):
    pairs = [(st.index(a), st.index(b)) for a, b in zip(labels, preds) if b in st]
    k, n = len(st), len(pairs)
    if n == 0 or k < 2:
        return None
    o = [[0] * k for _ in range(k)]
    for a, b in pairs:
        o[a][b] += 1
    ra = [sum(o[i]) for i in range(k)]
    cb = [sum(o[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2  # noqa: E731
    num = sum(w(i, j) * o[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else None


def full_metrics(rows, fruit, boot=1000, seed=0):
    st = STAGES[fruit]
    labels = [r["label"] for r in rows]
    preds = [r["pred"] if r["pred"] in st + ["unclear"] else "invalid" for r in rows]
    n = len(rows)
    correct = sum(a == b for a, b in zip(labels, preds))
    per = _f1s(labels, preds, st)
    macro = sum(v["f1"] for v in per.values()) / len(st)
    weighted = sum(v["f1"] * v["support"] for v in per.values()) / n if n else 0.0
    rng = random.Random(seed)
    bs = []
    for _ in range(boot if n else 0):
        idx = [rng.randrange(n) for _ in range(n)]
        p2 = _f1s([labels[i] for i in idx], [preds[i] for i in idx], st)
        bs.append(sum(v["f1"] for v in p2.values()) / len(st))
    bs.sort()
    d = [abs(st.index(b) - st.index(a)) for a, b in zip(labels, preds) if b in st]
    conf = {a: defaultdict(int) for a in st}
    for a, b in zip(labels, preds):
        conf[a][b] += 1
    lo, hi = wilson(correct, n)
    return dict(
        n=n, correct=correct, accuracy=correct / n if n else 0.0, acc_ci_low=lo, acc_ci_high=hi,
        balanced_accuracy=sum(per[c]["recall"] for c in st) / len(st),
        macro_f1=macro,
        macro_f1_ci_low=bs[int(0.025 * len(bs))] if bs else None,
        macro_f1_ci_high=bs[int(0.975 * len(bs)) - 1] if bs else None,
        weighted_f1=weighted,
        mae=sum(d) / len(d) if d else None, mse=sum(x * x for x in d) / len(d) if d else None,
        qw_kappa=qw_kappa(labels, preds, st),
        unclear=preds.count("unclear"), invalid=preds.count("invalid"),
        per_stage=per, confusion={a: dict(v) for a, v in conf.items()},
    )


def record(record_csv, rnd, model, fruit, split, results, note=""):
    rows = [json.loads(line) for line in open(results)]
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("source") or "all"].append(r)
    if len(groups) > 1:
        groups["all"] = rows
    out = {}
    new = not Path(record_csv).exists()
    with open(record_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for src in sorted(groups):
            m = full_metrics(groups[src], fruit)
            out[src] = m
            w.writerow(dict(
                round=rnd, model=model, fruit=fruit, split=split, source=src, note=note,
                **{k: (json.dumps(v) if isinstance(v, dict) else (round(v, 4) if isinstance(v, float) else v))
                   for k, v in m.items()},
            ))
    json.dump(out, open(Path(results).parent / "full_metrics.json", "w"), indent=1)
    for src, m in sorted(out.items()):
        print(f"{src:14s} n={m['n']:4d} acc={m['accuracy']:.3f} [{m['acc_ci_low']:.3f}-{m['acc_ci_high']:.3f}] "
              f"macroF1={m['macro_f1']:.3f} MAE={m['mae'] if m['mae'] is None else round(m['mae'], 3)} "
              f"MSE={m['mse'] if m['mse'] is None else round(m['mse'], 3)} kappa={m['qw_kappa'] and round(m['qw_kappa'], 3)}")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    record(a[0], a[1], a[2], a[3], a[4], a[5], a[6] if len(a) > 6 else "")

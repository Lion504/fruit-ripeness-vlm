"""Paired comparison of every public model with our two-pass answers on the same photos.

usage: compare_with_ours.py PREDS_DIR TWO_PASS_RESULTS.jsonl OUT.csv
PREDS_DIR holds <model>__<testset>.jsonl from predict.py; TWO_PASS_RESULTS is results.jsonl from
code/eval/two_pass_v7.py. Photos are matched by path, so both must come from the same machine.

Ours is right only if pass 1 names the right fruit AND the routed stage is right (for fruit naming: pass 1 only).
Where a public model has fewer stages, our "turning" label is merged the same way for both models.
For each row: accuracy of both with 95% Wilson ranges, the discordant counts and an exact two-sided McNemar p.
"""

import csv
import json
import math
import sys
from pathlib import Path

CARLOS_BANANA = {"Banana Unripe": "unripe", "Banana Ripe": "ripe", "Banana Overripe": "overripe"}
CARLOS_APPLE = {"Apple Unripe": "immature", "Apple Ripe": "mature", "Apple Overripe": "mature"}
DHAHLAN = {"unripe": "unripe", "ripe": "ripe", "overripe": "overripe"}
ARYAP2 = {"mentah": "unripe", "setengah-matang": "turning", "matang": "ripe", "terlalu-matang": "overripe"}
REITK = {"Class A": "unripe", "Class B": "turning", "Class C": "ripe", "Class D": "overripe"}
TCLEO = {"underripe": "unripe", "ripe": "ripe", "overripe": "overripe"}
PARSA = {"raw": "not_yet", "half_ripe": "not_yet", "ripe": "ready"}
DEANZA = {"ripe": "ready", "unripe": "not_yet"}
AFEEF = {"Green": "not_yet", "Breakers": "not_yet", "Turning": "not_yet", "Pink": "not_yet",
         "Light Red": "ready", "Red": "ready"}
OURS_SET = {"banana": "banana_test", "frame_small": "frame_small", "frame_cut": "frame_cut", "bg_new2": "bg_new2",
            "apple": "apple_test", "tomato": "tomato_test", "parsa_test": "tomato_parsa_test",
            "parsa_tomato_test": "tomato_parsa_test"}

# (public model id, preds key, test sets, label map, merges of our "turning", note)
RUNS = [
    ("Carlosmg23/fruit-ripeness-final", "carlos", ["banana"], CARLOS_BANANA, [None, "unripe", "ripe"],
     "3 banana stages; other classes count as wrong"),
    ("Carlosmg23/fruit-ripeness-final", "carlos", ["apple"], CARLOS_APPLE, [None],
     "Apple Unripe -> immature, Apple Ripe and Overripe -> mature; other classes wrong"),
    ("Dhahlan2000/banana_ripeness_level_detection", "dhahlan", ["banana"], DHAHLAN, [None, "unripe", "ripe"],
     "3 stages; rotten counts as wrong"),
    ("aryap2/kematangan-pisang-vit-b-32-100eph-224-v2.4", "aryap2", ["banana", "frame_small", "frame_cut", "bg_new2"],
     ARYAP2, [None], "4 stages from config id2label"),
    ("Reitk/banana-ripeness-classifier", "reitk", ["banana", "frame_small", "frame_cut", "bg_new2"], REITK, [None],
     "classes A-D in ripening order (source paper, by ripening day)"),
    ("TCleo/banana-maturity-mobile-vit-small", "tcleo", ["banana", "frame_small", "frame_cut", "bg_new2"], TCLEO,
     [None, "unripe", "ripe"], "3 stages"),
    ("ParsaVictor/tomato-ripeness-classifier-yolo26s", "parsa", ["tomato", "parsa_tomato_test"], PARSA, [None],
     "raw and half_ripe -> not_yet, ripe -> ready; rotten counts as wrong; their rotten test photos left out"),
    ("deanza1000/tomato-ripeness-resnet", "deanza", ["tomato", "parsa_test"], DEANZA, [None], "ripe / unripe"),
    ("afeefaazam/tomato-ripeness-detector", "afeef", ["tomato", "parsa_test"], AFEEF, [None],
     "detector; highest-confidence box, no box counts as wrong"),
]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(preds_dir, results, out_csv):
    ours = {}
    for line in open(results):
        d = json.loads(line)
        ours[(d["set"], d["path"])] = d
    rows = []

    def add(model, test, merge, pairs, note):
        n = len(pairs)
        t = sum(a for a, _ in pairs)
        o = sum(b for _, b in pairs)
        only_o = sum(b and not a for a, b in pairs)
        only_t = sum(a and not b for a, b in pairs)
        tl, th = wilson(t, n)
        ol, oh = wilson(o, n)
        rows.append(dict(model=model, test=test, turning_merged_into=merge or "", n=n,
                         theirs_correct=t, theirs_acc=round(t / n, 4), theirs_lo=round(tl, 4), theirs_hi=round(th, 4),
                         ours_correct=o, ours_acc=round(o / n, 4), ours_lo=round(ol, 4), ours_hi=round(oh, 4),
                         only_ours_right=only_o, only_theirs_right=only_t,
                         both_right=sum(a and b for a, b in pairs), both_wrong=sum(not a and not b for a, b in pairs),
                         mcnemar_p=f"{mcnemar(only_o, only_t):.2g}", note=note))

    for model, key, tests, m, merges, note in RUNS:
        for test in tests:
            f = Path(preds_dir) / f"{key}__{test}.jsonl"
            preds = [json.loads(line) for line in open(f)]
            for merge in merges:
                pairs = []
                for r in preds:
                    lab = r["label"]
                    if test == "parsa_tomato_test":
                        if lab == "rotten":
                            continue
                        lab = PARSA[lab]
                    o = ours[(OURS_SET[test], r["path"])]
                    our = o.get("stage_routed") if o.get("pass1") == o["fruit"] else None
                    their = m.get(r["raw"])
                    if merge:
                        lab, our, their = (merge if x == "turning" else x for x in (lab, our, their))
                    pairs.append((their == lab, our == lab))
                add(model, test.replace("parsa_tomato_test", "parsa_test"), merge, pairs, note)
    for fruit in ("banana", "apple", "tomato"):
        preds = [json.loads(line) for line in open(Path(preds_dir) / f"dima806__{fruit}.jsonl")]
        pairs = [(r["raw"] == fruit, ours[(OURS_SET[fruit], r["path"])].get("pass1") == fruit) for r in preds]
        add("dima806/fruit_vegetable_image_detection", f"fruit_naming_{fruit}", None, pairs,
            "fruit name only; ours = pass 1")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model'][:34]:34s} {r['test']:22s} {r['turning_merged_into']:7s} n={r['n']:4d} "
              f"theirs={r['theirs_acc']:.3f} ours={r['ours_acc']:.3f} {r['only_ours_right']}/{r['only_theirs_right']} "
              f"p={r['mcnemar_p']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

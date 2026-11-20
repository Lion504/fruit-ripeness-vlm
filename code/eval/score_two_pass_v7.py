"""Score a two_pass_v7.py run and add it to the round record.

usage: score_two_pass_v7.py RESULTS.jsonl RECORD_CSV ROUND MODEL_NAME

For each test set, two rows go to RECORD_CSV (full metrics, see code/metrics/full_metrics.py):
- split e2e_<set>: what the phone would show. Correct only if pass 1 names the right fruit AND the routed stage is
  right; a wrong fruit counts as an invalid answer, pass-1 'unclear' as unclear, pass-1 'none' as invalid.
- split stage_<set>: pass 2 with the true fruit's add-on (stage accuracy without routing errors).
Also writes pass1_summary.json next to RESULTS.jsonl: fruit-naming accuracy per set and over all fruit photos,
and the negatives (pass 1 must say none, or unclear for the dark / blurred ones).
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
from full_metrics import record  # noqa: E402


def main(results, record_csv, rnd, model):
    rows = [json.loads(line) for line in open(results)]
    by_set = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    summary = {}
    for name, rs in sorted(by_set.items()):
        p1 = Counter(r.get("pass1") for r in rs)
        ok = sum(r.get("pass1") == r["fruit"] for r in rs)
        summary[name] = dict(n=len(rs), pass1_correct=ok, pass1_accuracy=ok / len(rs), pass1_answers=dict(p1))
        if name == "negatives":
            continue
        fruit = rs[0]["fruit"]
        for kind in ("e2e", "stage"):
            out = []
            for r in rs:
                if kind == "stage":
                    pred = r.get("stage_true")
                elif r.get("pass1") == fruit:
                    pred = r.get("stage_routed")
                elif r.get("pass1") == "unclear":
                    pred = "unclear"
                else:
                    pred = f"wrong_fruit:{r.get('pass1')}"
                out.append(dict(label=r["label"], pred=pred, source=r["source"]))
            d = Path(results).parent / f"{kind}_{name}"
            d.mkdir(exist_ok=True)
            with open(d / "results.jsonl", "w") as f:
                for o in out:
                    f.write(json.dumps(o) + "\n")
            note = ("two-pass: pass-1 add-on then the named fruit's stage add-on; wrong fruit counts as invalid"
                    if kind == "e2e" else "pass 2 with the true fruit's add-on (no routing errors)")
            print(f"== {kind}_{name}")
            record(record_csv, rnd, model, fruit, f"{kind}_{name}", d / "results.jsonl", note)
    fr = [r for r in rows if r["set"] != "negatives"]
    summary["all_fruit_photos"] = dict(n=len(fr), pass1_correct=sum(r.get("pass1") == r["fruit"] for r in fr))
    summary["all_fruit_photos"]["pass1_accuracy"] = summary["all_fruit_photos"]["pass1_correct"] / len(fr)
    json.dump(summary, open(Path(results).parent / "pass1_summary.json", "w"), indent=1)
    print(json.dumps({k: (v["n"], round(v["pass1_accuracy"], 4)) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])

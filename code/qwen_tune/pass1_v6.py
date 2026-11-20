"""Step B: pass-1 fruit identification, phone form (strict one-field schema, thinking off, temperature 0, seed 42).
usage: pass1_v6.py OUT_DIR ENDPOINT LORA_ID|none SET=MANIFEST[:DEFAULT_EXPECTED] ...
Expected pass-1 answer per row: expected_produce if present; else 'none' for expected_status unsupported, 'unclear' for
unclear, else DEFAULT_EXPECTED (e.g. banana for the banana hard sets). Writes OUT_DIR/results.jsonl (one row per image,
saved before scoring) and OUT_DIR/score.json (computed from results.jsonl)."""

import json
import statistics
import sys
from pathlib import Path

from v6_common import PASS1_ENUM, PASS1_FORMAT, PASS1_PROMPT, PASS1_USER, ask, b64_of

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
ep, lid = sys.argv[2], sys.argv[3]
lora = None if lid == "none" else [{"id": int(lid), "scale": 1.0}]
done = {}
if (out / "results.jsonl").exists():
    for line in open(out / "results.jsonl"):
        d = json.loads(line)
        done[(d["set"], d["id"])] = d
f = open(out / "results.jsonl", "a")
rows = []
for spec in sys.argv[4:]:
    name, rest = spec.split("=", 1)
    path, _, default = rest.partition(":")
    for m in json.load(open(path)):
        mid = m.get("id") or m.get("image_id")
        if (name, mid) in done:
            rows.append(done[(name, mid)])
            continue
        exp = m.get("expected_produce") or {
            "unsupported": "none",
            "unclear": "unclear",
        }.get(m["expected_status"], default)
        try:
            raw, ms, _ = ask(
                ep,
                PASS1_PROMPT,
                PASS1_USER,
                PASS1_FORMAT,
                b64_of(m["path"]),
                lora=lora,
                max_tokens=30,
                temperature=0.0,
            )
            pred = json.loads(raw)["produce"]
        except Exception as e:
            raw, ms, pred = repr(e)[:100], 0, None
        d = {
            "set": name,
            "id": mid,
            "path": m["path"],
            "expected": exp,
            "stage": m.get("expected_ripeness"),
            "pred": pred,
            "ms": ms,
            "raw": raw,
        }
        f.write(json.dumps(d) + "\n")
        f.flush()
        rows.append(d)
    print(name, "done", flush=True)


def score(rows):
    res = {"n": len(rows), "correct": sum(r["pred"] == r["expected"] for r in rows)}
    res["accuracy"] = res["correct"] / len(rows)
    labels = PASS1_ENUM
    res["confusion"] = {
        e: {
            p: sum(1 for r in rows if r["expected"] == e and r["pred"] == p)
            for p in labels + [None]
        }
        for e in sorted({r["expected"] for r in rows})
    }
    ms = sorted(r["ms"] for r in rows if r["ms"])
    res["median_ms"] = statistics.median(ms) if ms else None
    by = {}
    for r in rows:
        if r["stage"]:
            c = by.setdefault(r["stage"], [0, 0])
            c[0] += r["pred"] == r["expected"]
            c[1] += 1
    res["by_stage_correct"] = by
    return res


sets = {}
for r in rows:
    sets.setdefault(r["set"], []).append(r)
sc = {k: score(v) for k, v in sets.items()}
fruit_rows = [r for r in rows if r["expected"] in ("banana", "apple", "tomato")]
sc["_all_fruit"] = score(fruit_rows) if fruit_rows else None
json.dump(sc, open(out / "score.json", "w"), indent=1)
for k, v in sc.items():
    if v:
        print(
            k,
            f"{v['correct']}/{v['n']}",
            round(v["accuracy"], 3),
            "median ms",
            v["median_ms"],
        )

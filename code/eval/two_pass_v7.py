"""End-to-end two-pass test with the current add-ons (run once; nothing here is used to choose anything).

usage: two_pass_v7.py OUT_DIR ENDPOINT PASS1_ID BANANA_ID APPLE_ID TOMATO_ID N_ADAPTERS NEGATIVES_JSON [NEG_ROOT ...]

The server holds the base model and all add-ons (started with --lora-init-without-apply). Every request lists
EVERY add-on, with scale 1 for the one it uses and 0 for the rest: llama-server keeps the previous request's
add-on on the slot when a request leaves the add-on field out, so nothing is left to the server.
Per photo: pass 1 (fruit naming add-on) -> banana / apple / tomato / none / unclear; if it names a fruit, pass 2
uses that fruit's stage add-on. Also records pass 2 with the TRUE fruit's add-on, so stage errors and routing
errors can be told apart.
Test sets: banana test (BANANA_SPLITS, split=test), banana frame_small / frame_cut / bg_new2 (TEST_LISTS_DIR),
apple test and tomato test (crops csv), ParsaVictor tomato test (TEST_LISTS_DIR/parsa_test.jsonl), negatives
(photos with no supported fruit: pass 1 must say none; dark / blurred ones: unclear).
Writes OUT_DIR/results.jsonl (one row per photo; resumable).
"""

import csv
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_tune"))
from paths import SPLITS_CSV, banana_path  # noqa: E402
from score_stage_v6 import rows_for  # noqa: E402
from v6_common import PASS1_FORMAT, PASS1_PROMPT, PASS1_USER, ask, b64_of, stage_format, stage_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LISTS = Path(os.environ.get("TEST_LISTS_DIR", ROOT / "data" / "lists"))


def jl(p):
    return [json.loads(line) for line in open(p)]


def neg_rows(manifest, roots):
    out, missing = [], []
    for m in json.load(open(manifest)):
        exp = m.get("expected_produce") or {"unsupported": "none", "unclear": "unclear"}[m["expected_status"]]
        p = Path(m["path"])
        if not p.is_file():
            hit = [q for r in roots for q in Path(r).rglob(p.name)]
            if not hit:
                missing.append(m["path"])
                continue
            p = hit[0]
        out.append(dict(set="negatives", path=str(p), fruit=exp, label=None, source="negatives"))
    return out, missing


def test_rows(neg_manifest, roots):
    rows = []
    for r in csv.DictReader(open(SPLITS_CSV)):
        if r["split"] == "test":
            rows.append(dict(set="banana_test", path=banana_path(r["path"]), fruit="banana", label=r["label"],
                             source=r.get("source", "bananaimagebd")))
    for name in ("frame_small", "frame_cut", "bg_new2"):
        rows += [dict(set=name, path=d["path"], fruit="banana", label=d["label"], source=name) for d in jl(LISTS / f"{name}.jsonl")]
    for fruit in ("apple", "tomato"):
        rows += [dict(set=f"{fruit}_test", path=p, fruit=fruit, label=lab, source=src) for p, lab, src in rows_for(fruit, "test")]
    rows += [dict(set="tomato_parsa_test", path=d["path"], fruit="tomato", label=d["label"], source="parsavictor_test",
                  their_class=d.get("their_class")) for d in jl(LISTS / "parsa_test.jsonl")]
    neg, missing = neg_rows(neg_manifest, roots)
    return rows + neg, missing


def main():
    out, ep = Path(sys.argv[1]), sys.argv[2]
    p1, ban, app, tom, n_ad = (int(x) for x in sys.argv[3:8])
    neg_manifest, roots = sys.argv[8], sys.argv[9:]
    ids = {"banana": ban, "apple": app, "tomato": tom}

    def lora(i):
        return [{"id": k, "scale": 1.0 if k == i else 0.0} for k in range(n_ad)]

    def stage(fruit, b):
        raw, ms, _ = ask(ep, stage_prompt(fruit), "What is the ripeness stage?", stage_format(fruit), b,
                         lora=lora(ids[fruit]), max_tokens=40, temperature=0.0)
        return json.loads(raw)["stage"], ms

    out.mkdir(parents=True, exist_ok=True)
    rows, missing = test_rows(neg_manifest, roots)
    json.dump(missing, open(out / "negatives_missing.json", "w"), indent=1)
    print("photos", len(rows), "negatives missing", len(missing), flush=True)
    done = set()
    if (out / "results.jsonl").exists():
        done = {(d["set"], d["path"]) for d in jl(out / "results.jsonl")}
    f = open(out / "results.jsonl", "a")
    for i, d in enumerate(rows):
        if (d["set"], d["path"]) in done:
            continue
        b = b64_of(d["path"])
        try:
            raw, d["ms_pass1"], _ = ask(ep, PASS1_PROMPT, PASS1_USER, PASS1_FORMAT, b, lora=lora(p1),
                                        max_tokens=30, temperature=0.0)
            d["pass1"] = json.loads(raw)["produce"]
            if d["fruit"] in ids:
                d["stage_true"], d["ms_stage_true"] = stage(d["fruit"], b)
            if d["pass1"] in ids:
                if d["pass1"] == d["fruit"]:
                    d["stage_routed"], d["ms_stage_routed"] = d["stage_true"], d["ms_stage_true"]
                else:
                    d["stage_routed"], d["ms_stage_routed"] = stage(d["pass1"], b)
        except Exception as e:  # noqa: BLE001
            d["err"] = repr(e)[:120]
        f.write(json.dumps(d) + "\n")
        f.flush()
        if (i + 1) % 200 == 0:
            print(i + 1, "/", len(rows), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

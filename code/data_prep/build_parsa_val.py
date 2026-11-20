"""Draw a ParsaVictor validation sample from their VALID split (rotten left out) for choosing epochs.

usage: build_parsa_val.py CROPS_ZIP OUT_DIR LIST_JSONL [N_RAW N_HALF N_RIPE SEED]   (default 350 350 300 36)
Crops are extracted to OUT_DIR/<class>/; the list has path, label (raw and half_ripe -> not_yet, ripe -> ready),
their_class and source=parsavictor_val.
"""

import json
import random
import sys
import zipfile
from pathlib import Path

MAP = {"raw": "not_yet", "half_ripe": "not_yet", "ripe": "ready"}


def main(zip_path, out_dir, list_path, n_raw=350, n_half=350, n_ripe=300, seed=36):
    z = zipfile.ZipFile(zip_path)
    out = Path(out_dir).resolve()
    want = {"raw": int(n_raw), "half_ripe": int(n_half), "ripe": int(n_ripe)}
    rng = random.Random(int(seed))
    rows = []
    for c, k in want.items():
        names = sorted(n for n in z.namelist() if n.startswith(f"cls_tomato/val/{c}/") and not n.endswith("/"))
        for n in sorted(rng.sample(names, k)):
            p = out / c / Path(n).name
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_bytes(z.read(n))
            rows.append(dict(path=str(p), label=MAP[c], their_class=c, source="parsavictor_val"))
    with open(list_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("val", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])

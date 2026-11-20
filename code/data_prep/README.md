# Data preparation

| Script | Makes |
| --- | --- |
| `build_banana_v7.py OLD_SPLITS_CSV OUT_CSV` | the banana split file: BananaImageBD (earlier splits kept) + BananaID (Ambon left out; all views of one banana in one split) + Prata Catarina crops (10% margin; burst photos of one banana in one split), 70/15/15 per stage |
| `build_tomato_parsa_train.py CROPS_ZIP OUT_DIR N_RAW N_HALF N_RIPE PER_BOX SEED` | extra tomato training crops from ParsaVictor's train split, one per photo or per tomato, with near-copies of any validation or test crop removed (64-bit difference hash, 6 bits or less). Final round: `2000 2000 2000 1 36` |
| `build_parsa_val.py CROPS_ZIP OUT_DIR LIST_JSONL` | the 1,000-crop ParsaVictor validation sample from their valid split (seed 36) |
| `build_banana_val_small.py OUT_DIR` | the small-in-frame validation set from banana validation photos (seed 35) |

Dataset folders: `BANANA_EXTRA_DIR` (BananaID and Prata Catarina, extracted), `PARSAVICTOR_DIR` (ParsaVictor test crops in
`test/<class>/`, validation samples in `val300/<class>/` and `val1000/<class>/`).

Apple and tomato field crops: `code/crops/`. Banana robustness test sets:
`code/robustness/`.

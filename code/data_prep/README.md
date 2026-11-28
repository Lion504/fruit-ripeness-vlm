# Data preparation

| Script | Makes |
| --- | --- |
| `build_banana_v7.py OLD_SPLITS_CSV OUT_CSV` | the banana split file: BananaImageBD (earlier splits kept) + BananaID (Ambon left out; all views of one banana in one split) + Prata Catarina crops (10% margin; burst photos of one banana in one split), 70/15/15 per stage |
| `build_tomato_parsa_train.py CROPS_ZIP OUT_DIR N_RAW N_HALF N_RIPE PER_BOX SEED` | extra tomato training crops from ParsaVictor's train split, one per photo or per tomato, with near-copies of any validation or test crop removed (64-bit difference hash, 6 bits or less). Final round: `2000 2000 2000 1 36` |
| `build_parsa_val.py CROPS_ZIP OUT_DIR LIST_JSONL` | the 1,000-crop ParsaVictor validation sample from their valid split (seed 36) |
| `build_banana_val_small.py OUT_DIR` | the small-in-frame validation set from banana validation photos (seed 35) |
| `build_neg_r39.py ROOT` | the non-fruit photos for the fruit-naming add-on (round 39): COCO 2017 validation, Open Images V7 and Fruits-360 crops, 1,500 train / 300 validation / 300 test, with `manifest_r39.csv` (source, photo id, licence, box). The script writes the cropped photos, `manifest_r39.csv` and `build_info_r39.json`; the lists below are derived from the manifest by me and are not written by the script. For training I saved its `train` rows as `neg_train_r39.csv` (columns `path,label`, label `none`) and gave that file to `train_lora_v6.py` as `NEG_EXTRA`; never pass the whole manifest, it also holds the validation and test rows. For scoring I saved the `val` and `test` rows as JSON lists (`id`, `path`, `expected_produce` set to `none`) and gave them to `pass1_v6.py` as the sets `neg_r39_val=FILE` and `neg_r39_test=FILE`. These small list files are not in the repository. ROOT holds `dl/` (COCO zips), `oi/` (Open Images csv files) and `f360/f360.zip` |
| `export_nonfruit_r39_list.py MANIFEST IMG_ROOT RESULTS_NEW RESULTS_PREV OUT_CSV` | writes `test_sets/test_nonfruit_r39.csv` (source-relative file names, box, licence, SHA-256, answers of both naming add-ons) |

Dataset folders: `BANANA_EXTRA_DIR` (BananaID and Prata Catarina, extracted), `PARSAVICTOR_DIR` (ParsaVictor test crops in
`test/<class>/`, validation samples in `val300/<class>/` and `val1000/<class>/`).

Apple and tomato field crops: `code/crops/`. Banana robustness test sets:
`code/robustness/`.

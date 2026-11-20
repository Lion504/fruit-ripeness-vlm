# Apple and tomato crops

The apple and tomato add-ons are trained and tested on crops of single fruits
cut from the original photos. These scripts make them:

- `prep_apple_mendeley.py`: Multi-Stage Apple Dataset (Mendeley gfcmdbvw65 v4),
  VIA polygon annotations to stage crops. Only original photos are used (the
  dataset's brightness / noise copies and resized folders are left out).
- `prep_tomato_openfield.py`: OpenField-BD Tomato Maturity (Zenodo 20176021),
  YOLO boxes to crops; the dataset's image-level splits are merged and split
  again by capture session, so photos from one session stay together.
- `prep_lib.py`: shared steps: crop each fruit, group exact and near-duplicate
  source photos, split by group per class, write the CSV.

Set `RIPEWISE_TRAINING_DIR` (see `paths.py`), `APPLE_DATASET_DIR` (the extracted apple dataset folder) and
`TOMATO_DATASET_DIR` (the extracted OpenField-BD `tomato_m_f` folder); crops are written under
`<training>/produce/crops/<fruit>/` with `train.csv`, `val.csv`, `test.csv`.
The test lists in `../../test_sets/` give every test crop's source photo and box.
Needs `pillow` and `imagehash`.

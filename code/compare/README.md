# Public-model comparison

- `predict.py MODEL TESTSET OUT.jsonl`: runs one public model over one test set
  with that model's own preprocessing and saves its raw predictions.
- `compare_public_models.py PREDS_DIR TWO_PASS_RESULTS.jsonl OUT.csv`: maps each
  model's labels to the model's own (for 3-stage models "turning" is merged the same
  way for both models) and runs an exact McNemar test against the two-pass
  answers on the same photos. Its output is `results/public_models_vs_this_model.csv`.

Paths: `COMPARE_MODELS_DIR` (downloaded models, one folder per model named
`<owner>__<name>`), `TEST_LISTS_DIR` (robustness and ParsaVictor test lists),
`PARSA_TEST` (ParsaVictor `cls_tomato/test` folder), plus the dataset variables
in `code/qwen_tune/paths.py`. The model weights are not redistributed here;
download them at these revisions:

| Model | Revision | Licence (as stated) |
| --- | --- | --- |
| [Carlosmg23/fruit-ripeness-final](https://huggingface.co/Carlosmg23/fruit-ripeness-final) | `d01889dbb7b592f3afab0edfda72332ece6e4bc2` | CC BY 4.0 |
| [Dhahlan2000/banana_ripeness_level_detection](https://huggingface.co/Dhahlan2000/banana_ripeness_level_detection) | `6c8397c7239fef80f6b6a26e61523930fb8546c3` | not stated |
| [ParsaVictor/tomato-ripeness-classifier-yolo26s](https://huggingface.co/ParsaVictor/tomato-ripeness-classifier-yolo26s) | `3dda8096bfa1ef3d5f9313330433573efd9d2e49` | AGPL-3.0 |
| [dima806/fruit_vegetable_image_detection](https://huggingface.co/dima806/fruit_vegetable_image_detection) | `02563e05a53735712aecff5b2ef4e57fd4fc626f` | Apache 2.0 |
| [aryap2/kematangan-pisang-vit-b-32-100eph-224-v2.4](https://huggingface.co/aryap2/kematangan-pisang-vit-b-32-100eph-224-v2.4) | `a8ea22ffb885bb9fd153160f044271f736d327d9` | not stated |
| [Reitk/banana-ripeness-classifier](https://huggingface.co/spaces/Reitk/banana-ripeness-classifier) (Space) | `70c50924d8e1ebe35e02910f2dbb374ed3b33af4` | not stated |
| [TCleo/banana-maturity-mobile-vit-small](https://huggingface.co/TCleo/banana-maturity-mobile-vit-small) | `1118af9672489cad3c59d226cc97fcaf4858527c` | GPL-2.0 |
| [deanza1000/tomato-ripeness-resnet](https://huggingface.co/deanza1000/tomato-ripeness-resnet) | `c633dab0a73b753fe43ff45fb9d95b8d8dd2c5de` | not stated |
| [afeefaazam/tomato-ripeness-detector](https://huggingface.co/afeefaazam/tomato-ripeness-detector) | `318c3e91578a748fa7020da80cbef00a357002e3` | MIT |

ParsaVictor's test crops come from the dataset
[ParsaVictor/tomato-ripeness-detection](https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection)
at revision `91481266b3674f4eec31e72310888321d0e102f2` (`tomato_ripeness_crops.zip`).

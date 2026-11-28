# Fruit ripeness grading with a 0.8B vision-language model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24417896.svg)](https://doi.org/10.5281/zenodo.24417896)

Training code, test lists, per-photo answers and every metric for the
[fruit ripeness add-ons for Qwen3.5-0.8B](https://huggingface.co/timo6677/qwen3.5-0.8b-fruit-ripeness-lora): four small LoRA add-ons
that name the fruit (banana, apple, tomato, none, unclear) and then grade its
ripeness stage, served with llama.cpp.

| Test set | Photos | Two-pass accuracy | Best public model I tested |
| --- | --- | --- | --- |
| Banana, 4 stages | 447 | 81.4% | 68.5% (3 stages only) |
| Apple, 3 stages | 304 | 94.4% | 32.2% |
| Tomato, field photos | 418 | 93.8% | 89.5% |
| Tomato, ParsaVictor conveyor test | 721 | 88.5% | 88.6% (their own test set) |
| Fruit naming | 2,210 | 99.5% | 64.7% (on 1,169) |
| Fruit naming, photos without a supported fruit (new test, round 39) | 300 | 99.7% | not tested |

The full write-up is in [REPORT.md](REPORT.md).

## Update, 28 November 2026 (round 39)

The first fruit-naming add-on named everyday objects as a fruit: a photo of a
desktop computer came back as "apple". The cause was in my training data: its
"none" examples were only background patches cut from fruit photos and plain or
noise images, so it had never seen an object or another fruit. I retrained it
with 1,500 extra non-fruit photos (COCO 2017, Open Images V7, Fruits-360) as
"none", added as the option `NEG_EXTRA` in `train_lora_v6.py` and built by
`code/data_prep/build_neg_r39.py`. Epoch 6 was chosen on validation.

On 300 new non-fruit test photos the answer "none" rose from 152 to 299 (a
peach is still called "apple"). Fruit naming on the 2,210 fruit photos is
unchanged at 2,199 right, with small shifts between sets (tomato 418 of 418
instead of 416, apple 303 of 304 instead of 304, five cut-off bananas called
"none" instead of one). The two-pass total over the 2,210 fruit photos is the same (1,961 right
for both add-ons), but single sets move by 1 or 2 photos, so the apple, field-tomato and
ParsaVictor numbers here and in REPORT.md changed by 0.1 to 0.5 points. The two small banana
robustness sets (80 photos each) moved by one photo each: small in the frame
69 to 70 right (86.3% to 87.5%), partly cut off 67 to 66 right (83.8% to 82.5%). On
Hugging Face only `addons/fruit_naming.gguf` changed (model repo commit
`e1474948ab85a440178421a6f8f1a44ae99edccf`, SHA-256
`9ce7e32d37e5a78f6a3264be4ce8a405f3825ae3176b7dbe6a697518fcc011d4`).

Limits: the new test photos come from the same three sources as the new
training photos (different photos, same kind of data), so 99.7% will not hold
for every real photo. The COCO annotations are CC BY 4.0 (COCO terms of use); each COCO photo keeps
its own Flickr licence, which is the one I recorded per photo.

## Intended use

Grading the ripeness of a single banana, apple or tomato in a photo (offline
apps, study and research). Not for food-safety or spoilage decisions, not for
commercial grading (the add-ons are non-commercial), and not for other fruits.
Details and limitations: the [model card](https://huggingface.co/timo6677/qwen3.5-0.8b-fruit-ripeness-lora).

## How to cite

See `CITATION.cff` (GitHub shows a "Cite this repository" button). The current
version is v1.0.1 (28 November 2026, retrained fruit-naming add-on). DOI for all
versions, which always points to the newest one:
[10.5281/zenodo.24417896](https://doi.org/10.5281/zenodo.24417896); DOI for v1.0.1:
[10.5281/zenodo.24626620](https://doi.org/10.5281/zenodo.24626620); DOI for v1.0.0:
[10.5281/zenodo.24417897](https://doi.org/10.5281/zenodo.24417897). Please also cite the
datasets listed at the end of `REPORT.md`.

## What is here

| Folder | Content |
| --- | --- |
| `code/qwen_tune/` | LoRA training (`train_lora_v6.py`, with `NEG_EXTRA` for extra non-fruit photos), prompts and request code (`v6_common.py`), stage scorer, fruit-naming scorer, GGUF conversion |
| `code/crops/` | cutting the apple and tomato fruit crops from the original photos |
| `code/data_prep/` | the banana split file, the ParsaVictor train and validation samples with the near-copy filter, the small-in-frame validation set, the non-fruit photos for fruit naming (`build_neg_r39.py`) and their test list (`export_nonfruit_r39_list.py`) |
| `code/robustness/` | how the banana robustness test sets were made (the images themselves are not published) |
| `code/eval/` | the two-pass test (`two_pass_v7.py`) and its scorer |
| `code/metrics/` | `full_metrics.py`: accuracy with 95% range, balanced accuracy, per-stage and macro F1 with bootstrap range, MAE and MSE in stages, quadratic weighted kappa, confusion matrix |
| `code/compare/` | running the public models on my test photos and the paired comparison; pinned model revisions in its README |
| `code/export_test_lists.py` | writes `test_sets/` from the local lists (dataset-relative paths only) |
| `test_sets/` | every test photo: dataset, file (relative to the dataset), crop box where a crop was used, label, SHA256 of the photo I scored. `answers_two_pass.csv` has my answer for every photo; `test_negatives.csv` lists the 91 photos without a supported fruit by number and SHA256 (not published). `test_nonfruit_r39.csv` lists the 300 new non-fruit test photos (source dataset, photo id, object box, licence of the photo, SHA256, my answers with the old and the new naming add-on) |
| `results/` | see below |

`results/`:

| File | Content |
| --- | --- |
| `rounds_clean.csv` | every epoch of every round (29-35), validation and test, full metrics |
| `rounds_clean_pass1.csv` | fruit-naming validation for every epoch |
| `rounds_r36.csv` | the final tomato round (validation and test) |
| `final_two_pass.csv` | the final two-pass run (fruit naming round 39): the numbers in the tables above |
| `rounds_r39_pass1.csv`, `rounds_e2e_r39.csv` | round 39: fruit-naming validation for every epoch (and the 300 non-fruit validation photos); the two-pass run with the round 33 and the round 39 naming add-on side by side |
| `rounds_base.csv`, `fruit_naming.csv` | the plain model without add-ons; fruit naming, plain model vs the add-ons (rounds 33 and 39) |
| `public_models_vs_this_model.csv` | the 9 public models vs this model's two-pass answers (fruit naming round 39) on the same photos, with exact McNemar tests |
| `compare.csv` | full metrics of the public models |
| `picks_clean.json`, `picks_r36.json`, `picks_provenance.csv` | the epoch choices, made on validation only, with how many seconds before the first test result each was written |
| `scoring_fault_check.csv` | answers changed by llama-server's prompt cache (see below) |

## Reproducing

Run order:

1. Download the datasets (citations and links at the end of REPORT.md) and the
   base model [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).
2. `pip install -r code/qwen_tune/requirements.txt`
3. Set the paths (see the top of `code/qwen_tune/paths.py`):
   `RIPENESS_TRAINING_DIR`, `BANANA_SPLITS`, `RIPENESS_BANANA_DATASET`,
   `RIPENESS_QWEN_BASE`, `RIPENESS_LLAMA_CPP`. The prompts in
   `code/qwen_tune/v6_common.py` must be kept word for word. (The published
   add-ons were trained with a first prompt sentence a few words longer; all
   test numbers use the prompts in this file, and the change moved 11 of 2,301
   test answers, none significantly.)
4. Make the crops and splits: `code/crops/prep_apple_mendeley.py`,
   `code/crops/prep_tomato_openfield.py`, then the scripts in `code/data_prep/`
   (see its README for arguments and seeds). For fruit naming also run
   `code/data_prep/build_neg_r39.py` (non-fruit photos, needs the COCO 2017 and
   Open Images V7 files and Fruits-360, see the top of the script).
5. Train each add-on and convert it to GGUF, for example the final tomato add-on:

   ```bash
   cd code/qwen_tune
   LOWRES=1 EPOCHS=6 EXTRA_TRAIN=parsa_train_r36/extra_train.csv OUT=runs/tomato python train_lora_v6.py tomato
   python lora_convert_v6.py runs/tomato/ep4 runs/tomato/ep4/ep4.gguf
   ```

   Settings of the final add-ons:

   | Add-on | Settings | Epoch |
   | --- | --- | --- |
   | Fruit naming | `pass1`, LOWRES=1 PARTIAL=1 EPOCHS=6, 1,200 ParsaVictor crops as EXTRA_TRAIN, 1,500 non-fruit photos as NEG_EXTRA (`neg_train_r39.csv`: the `train` rows of `manifest_r39.csv`, see `code/data_prep/README.md`) | 6 |
   | Banana | RANK=32 LOWRES=1 EPOCHS=10 | 10 |
   | Apple | defaults, EPOCHS=8 | 6 |
   | Tomato | LOWRES=1 EPOCHS=6, 6,000 ParsaVictor crops as EXTRA_TRAIN | 4 |

   The fruit-naming training set (fruit photos, background patches and
   generated images as "none", darkened or blurred fruit as "unclear", and with
   `NEG_EXTRA` the 1,500 extra non-fruit photos as "none") is built inside
   `train_lora_v6.py` (`load_rows`). The first release (round 33, epoch 4) was
   trained without `NEG_EXTRA`.
6. Score every epoch on validation (`score_stage_v6.py`, `pass1_v6.py`), choose
   the epoch, then run the tests once.
7. Serve the four add-ons with `llama-server` and run `code/eval/two_pass_v7.py`
   and `code/eval/score_two_pass_v7.py`.
8. Optional: `code/compare/predict.py` and `compare_public_models.py` for the
   public-model comparison.

**Important when scoring several add-ons on one llama-server:** send
`"cache_prompt": false` (done in `v6_common.ask`). llama-server reuses the
cached start of the prompt from the previous request even when that request
used another add-on; in my tests this changed up to 16 of 447 answers
(`results/scoring_fault_check.csv`).

## Licences

- Code: Apache 2.0 (`LICENSE`). To cite this work, see `CITATION.cff`.
- Add-ons: CC BY-NC 4.0 (the apple training data is marked non-commercial).
- Test lists point to the original datasets; no dataset photos are included.
  Dataset licences and citations are in REPORT.md. The non-fruit photos used
  for fruit naming are CC BY, CC BY-SA, CC BY-NC and CC BY-NC-SA licensed
  photos from COCO, Open Images and Fruits-360 (licence per photo in the test list).
- Licence mix of the 1,500 non-fruit training photos (round 39): 1,117 are
  CC BY, 77 CC BY-NC, 140 CC BY-SA (51 from COCO, 89 from Fruits-360) and 166
  CC BY-NC-SA. No photo is included in this repository or in the add-on file.
  Whether a trained add-on counts as an adaptation of its training photos is
  not settled, and I am not a lawyer. If it does, the ShareAlike terms of those
  306 photos would not fit CC BY-NC 4.0. The add-on stays CC BY-NC 4.0 for now;
  a retrain without the 306 ShareAlike photos would remove the question and
  has not been done.

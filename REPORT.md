# Grading fruit ripeness with a 0.8-billion-parameter vision-language model

## Summary

Four LoRA add-ons, 12-24 MB each, turn Qwen3.5-0.8B into a ripeness grader
for banana (4 stages), apple (3 stages) and tomato (2 stages). The first add-on
names the fruit; the second grades it. On held-out test photos from six public
datasets the two-pass system reaches 81.4% on banana, 94.7% on apple, 93.3% on
field tomatoes and 88.6% on conveyor-belt tomato crops, and names the fruit
correctly on 99.5% of 2,210 photos. Without the add-ons, the same model scores
31.8%, 45.4%, 81.1% and 57.0%.

I ran all 9 public fruit ripeness models I could find with downloadable
weights on the same photos. The add-ons are significantly more accurate than
each of them (exact paired McNemar test), with one exception: on ParsaVictor's
own tomato test set the two models are level (88.6% each, p = 1.0).

## 1. Why a small vision-language model

A ripeness grader that runs next to an app should be small, run offline, give
answers a program can read, and say when a photo is not a supported fruit or
too poor to judge. Most public ripeness models are single-fruit image
classifiers that always output one of their classes. A small vision-language
model can do all of this with one base model: switch a small add-on per task
and force the answer into a fixed JSON schema.

Qwen3.5-0.8B was the base. In 4-bit form (Q4_K_M) with its vision projector and
the four add-ons, the whole system is about 810 MB and answers a photo in about
0.27 s (median, two passes) on one consumer GPU.

## 2. Data

| Fruit | Dataset | Licence | Used as |
| --- | --- | --- | --- |
| Banana | BananaImageBD v2 | CC BY 4.0 | train / val / test |
| Banana | BananaID (Mendeley h6n5srjjyw); Ambon variety left out, its peel stays green | CC BY 4.0 | train / val / test |
| Banana | Banana Prata Catarina (Mendeley 7vb4djkbrc); 8 peel-colour degrees mapped 1-2 unripe, 3-5 turning, 6-7 ripe, 8 overripe | CC BY 4.0 | train / val / test |
| Apple | Multi-Stage Apple Dataset (Mendeley gfcmdbvw65 v4), crops from the masks | CC BY 4.0 on the record, CC BY-NC 4.0 in the article | train / val / test |
| Tomato | OpenField-BD Tomato Maturity (Zenodo 20176021); green and half-ripe = not_yet, fully ripe = ready | CC BY 4.0 | train / val / test |
| Tomato | ParsaVictor tomato ripeness crops (160 px conveyor crops); raw and half_ripe = not_yet, ripe = ready, rotten left out | CC BY 4.0 | their train split for training, their valid split for validation, their test split for testing |

Split rules:
- Photos of one banana (several views, or burst photos taken seconds apart)
  stay in one split.
- Added ParsaVictor training crops were compared with every test and validation
  crop using a 64-bit difference hash; anything within 6 bits was removed. A
  visual check showed the removed pairs were look-alikes rather than copies, so
  the filter errs on the safe side.

Robustness sets, made from BananaImageBD test photos: banana small in the
frame (80), partly cut off (80), on new backgrounds (160). A matching
small-in-frame validation set was made from validation photos only, so that
no choice was made on a test set.

## 3. Method

- **Add-ons:** LoRA on the language layers (attention of the 6 full-attention
  layers, feed-forward blocks of all 24 layers); vision encoder frozen. Rank 16,
  rank 32 for banana. The linear-attention layers were not adapted because the
  GGUF converter cannot export them.
- **Loss** on the answer value only, with inverse-frequency class weights
  (capped at 3).
- **Augmentation:** flips, crops (and, for fruit naming, crops that keep only
  part of the fruit), rotations filled from background photos, fruit pasted
  small onto backgrounds, low-resolution round trips (shrink to 48-160 px and
  back), colour jitter without hue shift, blur. Dark or heavily blurred photos
  are trained as "unclear".
- **Answers** are forced into a JSON schema (`{"stage": ...}` with an enum),
  temperature 0, thinking off.
- **Prompts:** the add-ons were trained with a first prompt sentence a few
  words longer than the published one. The final results below use the
  published prompts; compared with the training wording, 11 of 2,301 test
  answers changed, none of the differences significant.

## 4. Evaluation protocol

- Every epoch of every training round was scored on validation; the epoch was
  chosen by a rule fixed beforehand (validation macro F1, or a mean of two
  validation sets). The choice was written to a file before any test ran
  (`results/picks_*.json`; timing in `results/picks_provenance.csv`), and
  each test was run once.
- Metrics for every run: accuracy with 95% Wilson range, balanced accuracy,
  per-stage precision / recall / F1, macro F1 with 95% bootstrap range,
  weighted F1, MAE and MSE in stages, quadratic weighted kappa, confusion
  matrix, counts of "unclear" and invalid answers.
- Comparisons on the same photos use the exact two-sided McNemar test.
- Public models were run with their own preprocessing; labels were mapped to
  mine, and where a model has fewer stages my labels and answers were merged
  the same way.

### A scoring fault I found and fixed

Scoring the same add-on file twice gave different answers (up to 16 of 447).
The cause: llama-server keeps the computed start of the prompt (the system
prompt, the same for every photo) and reuses it for the next request, even when
the next request uses a different add-on. An add-on scored after another one on
the same server was therefore partly run with the other add-on. With
`"cache_prompt": false` the answers are identical across servers, restarts and
add-on orders (`results/scoring_fault_check.csv`). Every round was scored again with the fix, epochs were chosen
again, and all numbers in this report are from the fixed scoring.

## 5. Results

### Round by round (clean scoring, test sets, stage add-on with the true fruit, training prompts)

| Round | Change | Banana 447 | Banana small in frame | Tomato 418 | Tomato ParsaVictor 721 |
| --- | --- | --- | --- | --- | --- |
| 29 | first add-ons (BananaImageBD only for banana) | 69.4% | 66.3% | 94.0% | 75.0% (119 "unclear") |
| 30 | banana: + BananaID + Prata Catarina | 77.9% | 82.5% | | |
| 31 | low-resolution augmentation | 77.6% | 78.8% | 93.8% | 83.5% (39 "unclear") |
| 32 | tomato: + 2,500 ParsaVictor train crops | | | 94.3% | 87.4% |
| 33 | fruit naming: low-resolution and partial-fruit augmentation, ParsaVictor crops | | | | |
| 34 | banana: rank 32 | 82.1% | 58.8% | | |
| 35 | banana: rank 32 + low-resolution augmentation | 81.7% | 87.5% | | |
| 36 | tomato: + 6,000 ParsaVictor train crops | | | 93.5% | 88.9% |

Fruit naming on the ParsaVictor validation crops rose from 80.0% (round 29) to
97.7% (round 33); the round 29 add-on often called these small crops "none" or
"apple".

Round 34 shows why a small-in-frame validation set mattered: the larger banana
add-on was best on normal photos but failed on small bananas. Round 35, chosen
on both validation sets, kept most of the gain and fixed small bananas. On the
447 banana test photos round 35 beats round 30 on 30 photos and loses on 13
(p = 0.014); on Prata Catarina, 14 vs 3 (p = 0.013).

### Final system (two-pass)

| Test set | Photos | Accuracy | Macro F1 | MAE | Kappa |
| --- | --- | --- | --- | --- | --- |
| Banana | 447 | 81.4% [77.6-84.8] | 0.809 [0.773-0.843] | 0.20 | 0.90 |
| BananaImageBD part | 80 | 95.0% | 0.950 | 0.05 | 0.98 |
| BananaID part | 220 | 82.7% | 0.821 | 0.19 | 0.90 |
| Prata Catarina part | 147 | 72.1% | 0.695 | 0.29 | 0.81 |
| Banana small in frame | 80 | 86.3% | 0.874 | 0.13 | 0.94 |
| Banana cut off | 80 | 83.8% | 0.860 | 0.12 | 0.95 |
| Banana new backgrounds | 160 | 90.0% | 0.901 | 0.11 | 0.95 |
| Apple | 304 | 94.7% [91.6-96.7] | 0.933 | 0.05 | 0.97 |
| Tomato, field | 418 | 93.3% [90.5-95.3] | 0.888 | 0.06 | 0.78 |
| Tomato, ParsaVictor test | 721 | 88.6% [86.1-90.7] | 0.865 | 0.11 | 0.74 |

Fruit naming: 2,199 of 2,210 fruit photos right (99.5%); all 91 photos without
a supported fruit answered "none" or "unclear". Per-stage F1, banana: unripe
0.87, turning 0.67, ripe 0.84, overripe 0.86; apple: immature 0.99,
semi-mature 0.87, mature 0.94.

### Against public models

| Public model | Test | Theirs | This model | Only this model right / only theirs right | p |
| --- | --- | --- | --- | --- | --- |
| Dhahlan2000/banana_ripeness_level_detection | banana 447, turning merged into ripe | 66.4% | 89.0% | 112 / 11 | 3e-22 |
| TCleo/banana-maturity-mobile-vit-small | banana 447, turning merged into ripe | 68.5% | 89.0% | 111 / 19 | 5e-17 |
| Reitk/banana-ripeness-classifier | banana 447 | 53.9% | 81.4% | 146 / 23 | 4e-23 |
| aryap2/kematangan-pisang-vit-b-32 | banana 447 | 51.2% | 81.4% | 172 / 37 | 5e-22 |
| Carlosmg23/fruit-ripeness-final | banana 447, turning merged into ripe | 49.0% | 89.0% | 195 / 16 | 3e-40 |
| Carlosmg23/fruit-ripeness-final | apple 304 | 32.2% | 94.7% | 193 / 3 | 3e-53 |
| ParsaVictor/tomato-ripeness-classifier-yolo26s | tomato 418 | 89.5% | 93.3% | 36 / 20 | 0.044 |
| ParsaVictor/tomato-ripeness-classifier-yolo26s | their test 721 | 88.6% | 88.6% | 31 / 31 | 1.0 |
| deanza1000/tomato-ripeness-resnet | tomato 418 | 50.0% | 93.3% | 193 / 12 | 3e-43 |
| afeefaazam/tomato-ripeness-detector | tomato 418 | 59.8% | 93.3% | 154 / 14 | 6e-31 |
| dima806/fruit_vegetable_image_detection | fruit naming 1,169 | 64.7% | 99.8% | 411 / 0 | < 1e-40 |

Two public banana models report 90-98% on their own validation photos; on my
photos they score 51-54%. Accuracy on one dataset says little about photos
from another, which is why every model here, mine included, is tested on
several datasets.

## 6. Limitations

- Three fruits; tomato has only two stages.
- Banana "turning" (F1 0.67) and Prata Catarina photos (72.1%) are the weak
  spots. Stage boundaries differ between datasets.
- Half-ripe tomatoes called "ready" are most of the remaining tomato errors.
- Rejecting unsupported photos was tested on 91 photos only.
- Test photos come from public datasets, not from shop or kitchen phone photos.

## 7. Reproducing

Code, test lists (dataset-relative paths, crop boxes, SHA256 of every scored
photo), my answer for every test photo, all metrics and the pre-test choices
are in this repository. See README.md.

## Licences

Add-ons CC BY-NC 4.0 (the apple data is marked non-commercial in its article);
code Apache 2.0; base model Qwen3.5-0.8B Apache 2.0; datasets as listed in
section 2.

## Dataset citations

- Ferdaus, M. H., Prito, R. H., Rasel, A. A. S., Ahmed, M., Saykot, M. J. H.,
  Shanta, S. S., Akter, S., Das, A. C., Islam, M. M., Hasan, M., Ali, S. (2024).
  BananaImageBD: A Comprehensive Image Dataset of Common Banana Varieties with
  Different Ripeness Stages in Bangladesh (version 2). Mendeley Data.
  https://doi.org/10.17632/ptfscwtnyz.2. CC BY 4.0. Article: Data in Brief,
  https://doi.org/10.1016/j.dib.2024.111239
- Mutrofin, S., Fatichah, C., Yuniarti, A., Setiawan, E. (2025). BananaID: An
  image dataset of banana varieties and ripeness stages in Indonesia designed
  for classification (version 1). Mendeley Data.
  https://doi.org/10.17632/h6n5srjjyw.1. CC BY 4.0.
- Martins Neto, J. L. (2023). Dataset of Banana Prata Catarina Images Labeled in
  Eight Ripeness Stages (version 1). Mendeley Data.
  https://doi.org/10.17632/7vb4djkbrc.1. CC BY 4.0.
- Wang, D., Wang, B. (2026). A Multi-Stage, Pixel-Level Annotated Apple Dataset
  for Precision Agriculture Research (version 4). Mendeley Data.
  https://doi.org/10.17632/gfcmdbvw65.4. CC BY 4.0 on the record; the
  accompanying Data in Brief article (PMC13226782) states CC BY-NC 4.0.
- Rahim, U. F., Rahman, M. M. (2026). OpenField-BD-Tomato-Maturity-Detection
  Dataset. Zenodo. https://doi.org/10.5281/zenodo.20176021. CC BY 4.0.
- Karkooti, P. (2026). Unified Tomato Detection and Ripeness Dataset.
  https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection
  (code: https://github.com/ParsaVictor/tomato-ripeness-detection). CC BY 4.0.
  Merges Tomato Defects (Universitas Hasanuddin), tomato-sxsc7 (Dai Tran Van),
  tomato_count_project (canberk), tomato-v1 (papapeto),
  tomato-detection-sorting and tomato-classification.
- Base model: Qwen team, Qwen3.5-0.8B. https://huggingface.co/Qwen/Qwen3.5-0.8B.
  Apache 2.0.

# Metrics

`full_metrics.py RECORD_CSV ROUND MODEL FRUIT SPLIT RESULTS.jsonl [NOTE]` adds
one row per data source (and an "all" row) to RECORD_CSV:

- accuracy with a 95% Wilson range, balanced accuracy
- per-stage precision, recall and F1; macro F1 with a 95% bootstrap range; weighted F1
- MAE and MSE in stages, quadratic weighted Cohen's kappa
- confusion matrix; "unclear" and invalid answers count as wrong

All rows in `results/*.csv` were written with this script.

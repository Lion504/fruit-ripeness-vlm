# Two-pass test

- `two_pass_v7.py OUT_DIR ENDPOINT PASS1_ID BANANA_ID APPLE_ID TOMATO_ID N_ADAPTERS NEGATIVES_JSON [NEG_ROOT ...]`:
  sends every test photo through the server: pass 1 names the fruit, pass 2
  uses that fruit's add-on. It also records pass 2 with the true fruit's
  add-on, so naming errors and stage errors can be told apart. Every request
  lists all add-ons (scale 1 for the one used, 0 for the rest) and sets
  `cache_prompt: false`.
- `score_two_pass_v7.py RESULTS.jsonl RECORD_CSV ROUND MODEL_NAME`: writes the
  `e2e_<set>` rows (what an app would show; a wrong fruit counts as wrong) and
  the `stage_<set>` rows (true fruit's add-on) with `code/metrics/full_metrics.py`,
  and a fruit-naming summary.

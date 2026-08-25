# RankLab Reproducibility

This guide describes the frozen end-to-end analysis path for RankLab.

KuaiRand-Pure data and generated `runs/` artifacts are not committed.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
make setup
pytest
```

The completed M2.2 codebase passes 118 tests.

## Frozen protocol

Final M0 protocol SHA-256:

```text
9bd72ae2d3042dc9511734563b90b8668519b401b847ce218f706a465cc41a32
```

## Primary evaluation

```bash
python -m ranklab.evaluation.primary_runner \
  --training-data data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
  --data-dir data/raw/KuaiRand-Pure/data \
  --primary-dir runs/m0/primary_multiseed \
  --output-dir runs/m1/primary_raw
```

Frozen raw SHA-256:

```text
25f41d6380964e0eaf0f6746a8e4df221a3c62faa105ce33a3893d92f1d4301a
```

## Primary consolidation

```bash
python -m ranklab.evaluation.primary_results \
  --raw runs/m1/primary_raw/per_user_ndcg.csv.gz \
  --seed-mean runs/m1/primary_inference/per_user_seed_mean_ndcg.csv.gz \
  --within-manifest runs/m1/primary_inference/manifest.json \
  --cross-manifest runs/m1/cross_cell_contrasts/manifest.json \
  --output-dir runs/m1/primary_results
```

Frozen M1.4 hashes:

```text
b8939f44bc336ea6bac7691652b9d88b42975fac0e064453e4bc4720bc0b2065
6d137a8c1b5f68f717d406af9f9daa2d40ca849d19b3d79f4b8a80fe31971d0f
```

## Support sensitivities

Run `ranklab.evaluation.primary_sensitivity` for `shared_tabs` and `tab1`, then:

```bash
python -m ranklab.evaluation.sensitivity_inference \
  --shared-tabs-raw runs/m1/sensitivity_raw/shared_tabs/per_user_ndcg.csv.gz \
  --tab1-raw runs/m1/sensitivity_raw/tab1/per_user_ndcg.csv.gz \
  --output-dir runs/m1/sensitivity_inference
```

Frozen M1.6 manifest:

```text
8eac4c4f487c1dbd91388ee4cc2e8b740a5d696d7cb54d47bd898f0915d1e359
```

## Final M1 endpoint

```bash
python -m ranklab.evaluation.final_results \
  --primary-summary runs/m1/primary_results/summary.json \
  --primary-manifest runs/m1/primary_results/manifest.json \
  --sensitivity-manifest runs/m1/sensitivity_inference/manifest.json \
  --output-dir runs/m1/final_results
```

Frozen final M1 hashes:

```text
manifest.json
ba9334455134d86a14d2310c809c34152c0ce60f135bb62a765cdab9a1737c2f

summary.json
8735976eb8d5562de9728ddc98179f86ec0e5740f4ee09d8b8f04c0b1995ed33

FINAL_RESULTS.md
20ccd9083e71945581f7f2d9970404aa9b9287a30ba62a205dedcc083cf799bf
```

## Publication assets

```bash
python -m ranklab.reporting.publication \
  --final-manifest runs/m1/final_results/manifest.json \
  --final-summary runs/m1/final_results/summary.json \
  --final-markdown runs/m1/final_results/FINAL_RESULTS.md \
  --output-dir runs/m2/publication
```

This verifies the frozen final-M1 endpoint before generating tables and figures.

## Interpretation boundary

> **Winner identity is stable, but comparative margins are not.**

Logging-regime comparisons are offline evaluation contrasts, not causal
treatment-effect estimates. `tab=1` remains descriptive only. BPR and LightGCN
are not decisively separated under the frozen primary simultaneous intervals.

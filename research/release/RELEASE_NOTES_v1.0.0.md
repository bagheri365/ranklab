# RankLab v1.0.0

RankLab v1.0.0 is the first complete research release.

## Research question

> How stable is the offline model-selection decision when the same trained recommenders are evaluated under standard-policy versus randomized exposure?

## Frozen conclusion

> **Winner identity is stable, but comparative margins are not.**

Under the frozen protocol:

- Popularity is the decisive winner in all four primary regime × target cells.
- BPR and LightGCN are not decisively separated in any primary cell.
- Popularity's measured advantage is substantially larger under randomized exposure than under standard logging.
- Target definition also changes comparative margins, especially under randomized exposure.
- The pre-specified shared-tabs sensitivity preserves the qualitative result.
- The `tab=1` sensitivity also preserves the pattern but remains descriptive only.

These are offline evaluation contrasts, not causal treatment-effect estimates.

## Included research layers

### M0 — Benchmark integrity and protocol freeze

M0 freezes:

- KuaiRand-Pure provenance checks;
- logging-regime audit;
- target semantics;
- temporal split;
- training interaction and negative-sampling contract;
- BPR and LightGCN tuning protocols;
- frozen multi-seed checkpoints;
- evaluation support and scoring eligibility;
- exact NDCG semantics;
- statistical inference and multiplicity policy.

Final M0 protocol SHA-256:

```text
9bd72ae2d3042dc9511734563b90b8668519b401b847ce218f706a465cc41a32
```

### M1 — Frozen evaluation and inference

M1 includes:

- primary raw evaluation;
- within-cell paired user bootstrap inference;
- cross-regime and cross-target contrasts;
- support-sensitivity raw evaluation;
- sensitivity inference;
- final results consolidation.

Final M1 endpoint SHA-256 values:

```text
manifest.json
ba9334455134d86a14d2310c809c34152c0ce60f135bb62a765cdab9a1737c2f

summary.json
8735976eb8d5562de9728ddc98179f86ec0e5740f4ee09d8b8f04c0b1995ed33

FINAL_RESULTS.md
20ccd9083e71945581f7f2d9970404aa9b9287a30ba62a205dedcc083cf799bf
```

### M2 — Publication/reporting layer

M2 includes:

- frozen reporting contract;
- publication-ready CSV tables and SVG figures;
- public README results summary;
- reproducibility guide;
- paper-style Results + Discussion.

Final M2 narrative SHA-256:

```text
1905dc489908fdb746775b0b5b33e2666adc5225897f548979f3e5d69ce1ae29
```

## Reproducibility

See:

- `research/reporting/REPRODUCIBILITY.md`
- `research/reporting/M2.1_REPORTING_CONTRACT.md`
- `research/reporting/M2.2_PUBLICATION_TABLES_FIGURES.md`
- `research/reporting/M2.4_RESULTS_DISCUSSION.md`

Dataset files and generated `runs/` artifacts are not committed.

## Release boundary

v1.0.0 freezes the current RankLab research question, protocol, analysis, and reporting layer.

Future work that changes the estimand, candidate support, model family, behavioral target, inference rule, or research question should be treated as a new research version rather than silently altering the v1.0.0 result.

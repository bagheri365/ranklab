# RankLab

**RankLab studies how stable offline recommender model selection is when the same trained systems are evaluated under different logging regimes and behavioral targets.**

Primary question:

> How stable is the offline model-selection decision when the same trained recommenders are evaluated under standard-policy versus randomized exposure?

## Status

M0 and M1 are complete and frozen. M2 is the publication/reporting layer.

> **Winner identity is stable, but comparative margins are not.**

The frozen primary analysis finds Popularity as the decisive winner in all four
logging-regime × behavioral-target cells. BPR and LightGCN are not decisively
separated in any primary cell. Popularity's measured advantage is substantially
larger under randomized exposure than under standard logging, and the target
definition also changes the measured margin.

These are offline evaluation contrasts under different logged exposure
processes. They are **not causal treatment-effect estimates**.

### Primary macro NDCG@10

| Logging regime | Target | Popularity | BPR | LightGCN |
| --- | --- | ---: | ---: | ---: |
| standard | `is_click` | **0.716524** | 0.676985 | 0.676862 |
| standard | `long_view` | **0.648404** | 0.601746 | 0.601590 |
| randomized | `is_click` | **0.432459** | 0.343522 | 0.343527 |
| randomized | `long_view` | **0.370164** | 0.265048 | 0.265314 |

Support robustness preserves the same qualitative pattern under both the
pre-specified shared-tabs sensitivity and the descriptive `tab=1` sensitivity.
`tab=1` is not assigned a semantic UI meaning because the public KuaiRand
documentation does not map every numeric tab ID to a semantic label.

See:

- `research/reporting/M2.1_REPORTING_CONTRACT.md`
- `research/reporting/M2.2_PUBLICATION_TABLES_FIGURES.md`
- `research/reporting/REPRODUCIBILITY.md`

## Primary study

```text
same 3 trained systems
x 2 logging regimes
x 2 predeclared behavioral targets
x 1 primary metric
x 1 primary K
```

Frozen model families for M1:

1. popularity / engagement baseline
2. BPR matrix factorization
3. LightGCN

The same fitted checkpoints per frozen training seed will be reused across every regime × target condition.

## Repository map

- `src/ranklab/` — implementation
- `configs/` — model, benchmark, and experiment parameters
- `research/` — protocol decisions, target cards, audits, and invalidations
- `runs/` — generated machine-readable run artifacts (gitignored by default)
- `reports/` — curated human-readable figures/tables/results
- `tests/` — unit and integration tests, including ranking metric fixtures

## First milestone: M0

M0 must verify KuaiRand-Pure logging semantics, target validity, temporal comparability, training supervision, candidate-set semantics, exact metric behavior, matched support, uncertainty procedures, and protocol integrity.

The final M0 protocol is frozen. Historical milestone notes below are retained as an audit trail; current status is summarized above and in `research/reporting/REPRODUCIBILITY.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make test
ranklab --help
```

Dataset files are not committed. See `data/README.md`.

## M0.1 — data provenance

After downloading and extracting the official KuaiRand-Pure archive, run:

```bash
make audit-data
```

This records SHA-256 hashes for the six expected CSVs and validates the minimum documented log headers. It is a provenance/schema check only; it does not freeze target, UI-scenario, candidate-set, or ranking-unit semantics.

## M0.2 — regime and target audit

After M0.1 provenance succeeds, run:

```bash
make audit-regimes
```

This produces `runs/m0/regime_audit.json` with descriptive statistics for the three interaction logs and user/item overlap between the two Apr 22–May 8 evaluation regimes. It does not freeze target semantics or M1 conclusions.

### M0.3 — shared-support / scenario audit

After M0.2, inspect increasingly comparable evaluation slices without freezing a matched-support rule:

```bash
make audit-support
```

This writes `runs/m0/support_audit.json`. The `tab=1` slice is descriptive only: the public KuaiRand documentation identifies `tab` as a scenario code but does not publicly map every numeric code to a semantic UI name.

### M0.4 — target semantics / label-quality audit

After the support/scenario audit, verify the proposed behavioral targets before freezing them:

```bash
make audit-targets
```

This writes `runs/m0/target_audit.json` with duration-stratified target prevalence, target co-occurrence, anomaly counts, and a direct check of the documented `long_view` rule. `is_click` remains UI-dependent in the official documentation, so RankLab does not reconstruct it from play time without an authoritative mapping from `tab` to UI type.

### M0.5 — training interaction / temporal split audit

Before training any primary model, audit candidate interaction definitions that are distinct from `is_click` and `long_view`, including a downstream-engagement composite (`is_profile_enter` or a positive social action), together with a leakage-safe validation split drawn only from the standard-history period:

```bash
make audit-training
```

This writes `runs/m0/training_audit.json`. The default three-day trailing validation slice is descriptive only; M0 must still freeze the final interaction definition, cutoff, negative policy, validation objective, and model losses before any primary training run.


### M0.6 training-contract freeze candidate

After the training-signal density audit, RankLab tests a click-based implicit-feedback contract before freezing it:

```bash
make audit-training-contract
```

The audit checks pair collapse, positive/negative coverage, 1:1 logged-negative feasibility, and validation ranking-unit support without inspecting the Apr 22–May 8 test labels.

### M0 training contract status

The training subsection is now frozen from M0.6b: standard-policy click pairs from Apr 9–16, Apr 17–21 validation, same-user logged negatives sampled uniformly with replacement, and a deterministic clicked-pair popularity baseline. The **overall M0 protocol remains UNFROZEN** until evaluation, support, and uncertainty decisions are complete.

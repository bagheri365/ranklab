# RankLab

**RankLab studies how stable offline recommender model selection is when the same trained systems are evaluated under different logging regimes and behavioral targets.**

Primary question:

> How stable is the offline model-selection decision when the same trained recommenders are evaluated under standard-policy versus randomized exposure?

## Status

**M0 — Benchmark Integrity: scaffolded, not frozen.**

The repository intentionally does **not** contain primary M1 results yet. Dataset semantics, target definitions, the logged ranking unit, exact NDCG semantics, training supervision, matched-support rules, and statistical procedures must be audited and frozen in M0 before comparative M1 evaluation.

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

The primary protocol is not considered frozen until `research/protocol_frozen_m0.yaml` is complete and its SHA-256 is computed externally. Every valid M1 manifest must reference that hash.

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

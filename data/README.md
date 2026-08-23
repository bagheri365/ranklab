# Data

RankLab's primary benchmark is **KuaiRand-Pure**. Raw and derived benchmark data are intentionally excluded from Git.

## Official source

Use the official Zenodo record:

- https://zenodo.org/records/10439422
- archive: `KuaiRand-Pure.tar.gz`
- published MD5: `0820331067a3784d9691136f772b35a7`

The upstream KuaiRand repository documents the archive and expected Pure file layout:

- https://github.com/chongminggao/KuaiRand

## Local layout

```text
data/
├── raw/
│   └── KuaiRand-Pure/
│       └── data/
│           ├── log_random_4_22_to_5_08_pure.csv
│           ├── log_standard_4_08_to_4_21_pure.csv
│           ├── log_standard_4_22_to_5_08_pure.csv
│           ├── user_features_pure.csv
│           ├── video_features_basic_pure.csv
│           └── video_features_statistic_pure.csv
├── interim/
└── processed/
```

## Acquisition

Download the archive from the official Zenodo record, place it outside Git tracking, verify its MD5, and extract it under `data/raw/`.

Example on macOS/Linux:

```bash
curl -L \
  https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o data/raw/KuaiRand-Pure.tar.gz

md5 data/raw/KuaiRand-Pure.tar.gz  # macOS
# md5sum data/raw/KuaiRand-Pure.tar.gz  # Linux

tar -xzf data/raw/KuaiRand-Pure.tar.gz -C data/raw/
```

Expected archive MD5:

```text
0820331067a3784d9691136f772b35a7
```

Then run:

```bash
ranklab audit-data --data-dir data/raw/KuaiRand-Pure/data
```

The audit checks the expected file inventory, computes SHA-256 hashes for the six CSVs, and verifies the minimum documented log columns. It does **not** freeze target, scenario, ranking-unit, or candidate-set semantics.

See `research/audits/logging_regimes/M0.1_DATA_PROVENANCE.md` for the provenance record and unresolved M0 questions.

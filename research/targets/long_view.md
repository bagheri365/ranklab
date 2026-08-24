# Target card — `long_view`

Status: **FROZEN**

- Raw source field(s): `long_view`, with `play_time_ms` and `duration_ms` used only to audit the published construction
- Raw scale: binary `{0,1}`
- Transformation at row level: none for evaluation; use the dataset-provided `long_view`
- Published construction audited in M0: positive when `play_time_ms >= duration_ms` for videos of duration at most 18,000 ms, otherwise positive when `play_time_ms >= 18,000`
- Evaluation pair transformation: for a retained `(user_id, video_id)` pair, relevance is 1 if any retained exposure row has `long_view=1`, otherwise 0
- Threshold: dataset-defined duration-dependent rule described above
- Missingness: core evaluation audits found no missingness in the required target/support fields
- Temporal availability: available in the frozen training, validation, and evaluation logs
- Training role: none; no model or hyperparameter selection uses `long_view`
- Evaluation role: target B, evaluated on the same frozen checkpoints as `is_click`
- Duration dependence: explicit and material by construction
- Scenario dependence: observed prevalence varies strongly with logging/exposure regime and scenario composition
- Known confounders: exposure policy, scenario mix, video duration, and opportunity to accumulate watch time
- Meaning: the dataset's duration-thresholded long-view behavioral signal
- Does **not** mean: a duration-independent preference label or a signal used to tune RankLab models
- Audit reference: `research/audits/targets/M0.4_TARGET_SEMANTICS_AUDIT.md`
- Evaluation-semantics reference: `research/audits/comparability/M0.19_EVALUATION_SEMANTICS_FREEZE.md`

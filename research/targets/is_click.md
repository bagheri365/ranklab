# Target card — `is_click`

Status: **FROZEN**

- Raw source field(s): `is_click`
- Raw scale: binary `{0,1}`
- Transformation at row level: none; use the dataset field as provided
- Evaluation pair transformation: for a retained `(user_id, video_id)` pair, relevance is 1 if any retained exposure row has `is_click=1`, otherwise 0
- Threshold: none beyond the raw binary field
- Missingness: core evaluation audits found no missingness in the required target/support fields
- Temporal availability: available in the frozen training, validation, and evaluation logs
- Training role: the sole training-positive signal for Popularity/BPR/LightGCN
- Evaluation role: target A, evaluated independently from `long_view`
- Scenario dependence: the dataset documentation states that click semantics depend on interface scenario; in two-column UI it is click, while in single-column UI it corresponds to valid play
- Known confounders: logging/exposure regime and scenario composition materially change observed prevalence
- Meaning: the dataset's binary `is_click` behavioral signal under the logged interface/exposure process
- Does **not** mean: an exposure-independent or universally identical notion of click across all scenarios
- Audit reference: `research/audits/targets/M0.4_TARGET_SEMANTICS_AUDIT.md`
- Evaluation-semantics reference: `research/audits/comparability/M0.19_EVALUATION_SEMANTICS_FREEZE.md`

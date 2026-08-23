# RankLab — Project Plan

## Project thesis

**RankLab studies how stable offline recommender model selection is when the same trained systems are evaluated under different logging regimes and behavioral targets.**

Primary research question:

> **How stable is the offline model-selection decision when the same trained recommenders are evaluated under standard-policy versus randomized exposure?**

Secondary research question:

> **Does model selection remain stable when the behavioral target changes?**

Explanatory question, activated only if instability is observed:

> **Which user, item, and representation properties explain model-selection reversals?**

The project is intentionally **not** a “Netflix clone,” generic recommender demo, dataset showcase, industrial-scale simulation, or architecture leaderboard.

The first study is deliberately narrow:

```text
same trained models
x
2 logging regimes
x
2 predeclared behavioral targets
x
1 primary decision depth
```

Everything else is conditional on what that study finds.

---

# 1. Core research principles

1. **Freeze the benchmark before model comparison.**
2. **Train the same core systems once for the primary study; vary evaluation regime, not training distribution.**
3. **Predeclare exactly two primary behavioral targets.**
4. **Predeclare one primary decision depth K.**
5. **Treat standard-policy and randomized exposure as different logging regimes, not “biased vs truth.”**
6. **Audit regime comparability before interpreting any reversal.**
7. **Do not promote side features, reranking, or additional models until a concrete unresolved result requires them.**
8. **Separate evaluation uncertainty, training uncertainty, and protocol sensitivity.**
9. **Report practical effect sizes as well as confidence intervals.**
10. **Use language no stronger than the logged data support.**
11. **Preserve null and invalidated results.**

Governing rule:

> **RankLab does not add explanatory machinery until the primary model-selection stability experiment produces a phenomenon worth explaining.**

---

# 2. Dataset strategy

## Primary benchmark

Use **KuaiRand-Pure**.

It is selected because it provides:

- standard recommendation exposure logs
- randomized exposure logs
- timestamps
- multiple behavioral feedback signals
- user and item attributes
- a scale appropriate for CPU-based experimentation on a Mac

## Logging-regime interpretation

RankLab must not describe randomized exposure as automatically “unbiased ground truth.”

The project must verify exactly how randomized exposure was generated and use terminology consistent with that mechanism.

Preferred language:

```text
standard-policy exposure
randomized exposure
```

Avoid:

```text
biased data
unbiased truth
```

unless the dataset design genuinely supports that claim.

## Default temporal design

Use the dataset's natural logging periods as the presumptive primary split:

```text
TRAIN
Apr 8–21:
standard-policy history

TEST A
Apr 22–May 8:
standard-policy exposure

TEST B
Apr 22–May 8:
randomized exposure
```

Validation should come from an earlier standard-policy holdout inside the training-era history.

This design is preferred because Test A and Test B occupy the same calendar period, reducing temporal confounding.

M0 may alter exact boundaries only if the data audit reveals a concrete problem.

## Primary comparability requirement

Before interpreting any standard-vs-random reversal, M0 must audit whether the two regimes differ in:

- user support and overlap
- item support and overlap
- time windows
- history-length distributions
- item-popularity distributions
- target prevalence
- feature availability
- exposure counts
- candidate-set structure

If differences are substantial, use matched/restricted analyses such as:

```text
shared users
shared time window
shared item support
stratified estimates
```

The primary conclusion must distinguish **logging-regime effects** from **population-composition effects**.

## External-validity dataset

A second dataset is optional and evidence-driven.

Preferred options:

- **KuaiRec** for evaluation-bias / near-complete-observation analysis
- **MIND-small** for impression-level ranking with rich content features

A second dataset enters only if the primary conclusion requires replication.

---

# 3. Benchmark contract

Before comparative test evaluation, freeze:

- KuaiRand-Pure version/checksum
- exact semantics of standard vs randomized exposure
- two primary behavioral targets
- target-definition cards
- training-data source policy
- primary training supervision / positive-interaction definition
- treatment of non-positive exposures and negative-sampling policy
- frozen training loss/objective and validation objective
- exact three-model set and comparable hyperparameter-search budget
- model-checkpoint identity across all four M1 conditions for every frozen seed
- temporal split
- user/item eligibility rules
- logged-exposure evaluation protocol, including ranking unit, candidate construction, aggregation, weighting, and exact NDCG semantics
- deterministic metric-fixture tests for the frozen NDCG implementation
- one primary decision depth K
- primary endpoint(s)
- secondary/exploratory endpoint hierarchy
- model-selection rule
- decisive-winner / no-decision algorithm
- minimum practical-effect threshold for declaring preference and how it is chosen without M1 test outcomes
- bootstrap resampling unit, replicate count, CI method, confidence level, multiplicity policy, and cross-seed aggregation rule
- uncertainty-estimation plan
- regime-comparability audit
- frozen matched-support restriction rule and retention-reporting requirements
- required native-regime and matched-support companion analyses
- direct pairwise-margin contrast estimands across logging regime and behavioral target, including their clustered uncertainty procedure
- one authoritative frozen M0 protocol artifact and its SHA-256 hash
- invalidation rules

Every experiment should emit a compact manifest similar to:

```json
{
  "experiment_id": "ranklab-m1-014",
  "dataset": "kuairand-pure",
  "dataset_sha256": "...",
  "protocol_sha256": "...",
  "train_policy": "standard_only",
  "train_target": "frozen_primary_interaction",
  "train_objective": "...",
  "model_checkpoint_sha256": "...",
  "seed_set": [365, 366, 367],
  "eval_policy": "randomized",
  "target": "target_a",
  "primary_k": 10,
  "config": "configs/experiments/m1.yaml",
  "seed": 365,
  "git_commit": "...",
  "result_sha256": "..."
}
```

---

# 4. Feedback, exposure, and temporal evaluation protocol

## 4.1 Exactly two primary behavioral targets

Default targets for M0:

### Target A — `is_click`

Use the dataset's provided click / valid-play label.

M0 must verify its semantics across UI scenarios and may restrict evaluation to comparable scenarios if necessary.

### Target B — `long_view`

Use the dataset's provided long-view label.

This is preferred over raw watch time because raw duration can strongly confound watch-time comparisons.

M0 must still verify:

- prevalence
- duration dependence
- scenario dependence
- missingness
- whether the label semantics are consistent enough for the primary study

These defaults may be replaced only if M0 reveals a concrete data-quality or interpretation problem.

Do not select replacement targets based on which produce the most dramatic reversal.

### Target rejection criteria

A proposed primary target must be rejected or restricted during M0 if any of the following hold:

- its semantics differ materially across included UI scenarios
- missingness differs substantially across logging regimes
- prevalence is too low for stable evaluation under the frozen uncertainty plan
- the label is mechanically dominated by content duration or another nuisance variable in a way that undermines interpretation
- the target cannot be computed identically across both evaluation regimes

If restriction to comparable scenarios resolves the issue without materially changing the study population, document and freeze that restriction before M1.

## 4.2 Target-definition cards

Each target must document:

```text
name
raw source field(s)
transformation
threshold if any
prevalence
class imbalance
missingness
temporal availability
known confounders
what it means
what it does NOT mean
```

If watch-based targets are used, explicitly analyze duration dependence.

For example:

- raw watch seconds may favor long content
- completion may favor short content
- watch ratio changes the normalization assumption

The chosen target must be frozen before comparative test evaluation.

## 4.3 Primary train/evaluate design

For the core RankLab study:

```text
TRAIN:
standard-policy history only

EVALUATE A:
standard-policy test exposures

EVALUATE B:
randomized test exposures
```

This isolates evaluation-regime sensitivity better than simultaneously changing both training and evaluation distributions.

### Primary training contract

All three primary systems are trained exactly once from standard-policy history using one frozen interaction definition. The training interaction definition is distinct from the two primary evaluation targets.

M0 must freeze:

- positive-interaction definition
- treatment of non-positive exposures
- negative-sampling policy
- loss / training objective for each model family
- training cutoff
- validation objective
- hyperparameter search space and comparable search budget
- seed policy

The two evaluation targets (`is_click`, `long_view`) may not be used to retrain or retune separate versions of the primary systems. For each frozen training seed, the exact same fitted checkpoint must be evaluated in all four M1 conditions.

Training on mixed or randomized logs, or target-specific retraining, is a later experiment only if the primary result requires it.

## 4.4 Feature provenance by logging source

Every feature family must declare whether it may use:

```text
standard logs only
random logs only
both
```

For the core study, features intended to represent a system trained under standard historical logging should be built from standard-policy history only.

Randomized-evaluation labels must not leak into training features.

## 4.5 Temporal protocol

Presumptive primary protocol:

```text
Apr 8–21:
training / validation from standard-policy history

Apr 22–May 8:
standard-policy test exposure
randomized test exposure
```

Validation must be drawn from the earlier standard-policy period without touching the final test dates.

Freeze:

- exact training/validation cutoff
- shared test calendar window
- user/item eligibility
- item-availability rules
- scenario restrictions if required

Do not select a temporal protocol based on which gives more interesting reversals.

## 4.6 Candidate evaluation

### Primary mode

Use **logged-exposure evaluation** for the central stability study.

This is the default because it preserves the meaning of the observed logging regimes.

### Logged-evaluation contract

Before M1, M0 must freeze the exact evaluation unit and scoring procedure from the verified KuaiRand logging semantics:

- ranking / query unit (for example, impression, request, user-time slice, or another dataset-supported grouping)
- candidate-set construction
- relevance-label construction for each frozen target
- duplicate-exposure handling
- minimum candidate-set size
- treatment of candidate sets with size `< K`
- aggregation unit for the primary metric
- weighting policy across users / ranking units
- treatment of users or items unseen during training
- whether any test-time negative sampling is permitted
- treatment of ranking units containing zero relevant items
- deterministic model-score tie-breaking policy
- exact NDCG gain function
- exact NDCG discount function
- relevance scale for each frozen target, including whether relevance is binary or graded

The frozen primary-metric implementation must be deterministic and covered by unit tests against hand-calculated ranking fixtures, including at least: a perfect ranking, a reversed ranking, a tie case, a ranking unit with fewer than `K` candidates, and a zero-relevance ranking unit.

Primary evaluation must use observed logged candidates only. Unexposed catalog items are not silently added to primary candidate sets. Test labels and randomized-exposure information must not affect candidate construction, model fitting, hyperparameter selection, or model scoring.

If the dataset does not support a defensible common ranking unit across the two regimes, M0 must narrow or invalidate the primary design rather than inventing an artificial comparison.

### Secondary mode

Full-catalog evaluation is permitted only for a later candidate-generation question.

It is not a co-equal primary benchmark.

---

# 5. Frozen hypotheses

## H1 — Logging-regime stability

> The identity of the selected model may change when the same trained systems are evaluated under standard-policy versus randomized exposure.

A reversal is not assumed.

## H2 — Behavioral-target stability

> The identity of the selected model may change between the two predeclared behavioral targets.

Again, a reversal is not assumed.

## H3 — Conditional instability

> If reversals occur, their frequency or magnitude may vary with user history, item history/popularity, or regime composition.

This hypothesis is explanatory and is tested only after a primary reversal is established.

## H4 — Personalized-model robustness

> Any important reversal should persist beyond a trivial popularity-vs-personalization comparison.

M1 therefore includes two competent personalized systems alongside the non-personalized baseline from the start. The third system is not conditional on observing a reversal.

## H5 — Falsification criterion

The central instability thesis is **not supported** if:

- the same decisive model is selected across both logging regimes for both primary targets, or conditions are consistently inconclusive rather than reversals,
- pairwise differences are small relative to uncertainty and the frozen practical-effect threshold,
- apparent native-regime reversals do not persist under the required matched-support companion analysis,
- and no practically meaningful decisive reversal appears under the predeclared primary analysis.

A null result is a valid project outcome.

---

# 6. Model-selection stability, endpoints, and uncertainty

## 6.1 Formal model-selection object

For each primary condition:

```text
condition =
(logging regime, behavioral target, primary K)
```

select the preferred model using the frozen primary metric and selection rule.

### Decisive-winner rule

For each primary condition, use the following frozen algorithm:

1. Evaluate every stochastic model under the same three frozen training seeds. The deterministic baseline has one fitted state and is reused unchanged.
2. Compute the frozen primary metric separately for each model/seed, then use the mean across the three frozen seeds as the model's primary point estimate.
3. Rank the three systems by those seed-aggregated point estimates and treat the highest-scoring system as the candidate winner.
4. For each candidate-winner-versus-alternative comparison, perform a paired **user-clustered bootstrap**. Within each bootstrap replicate, resample users with replacement, recompute the metric difference separately for each frozen seed, then average those seed-specific differences to obtain one seed-aggregated bootstrap difference.
5. Use the predeclared confidence level, CI method, replicate count, and multiplicity policy frozen in M0.
6. Check training-seed stability separately: the candidate winner must not be outperformed by the same alternative in any frozen seed. Any seed-level ordering reversal against an alternative makes that pair training-seed-unstable.
7. Declare a decisive winner only if, against **every** alternative, the seed-aggregated point-estimate difference is at least the frozen minimum practical-effect threshold `delta`, the seed-aggregated bootstrap comparison satisfies the frozen uncertainty criterion, and the pair is training-seed-stable.
8. Otherwise report `no decisive winner`.

M0 must freeze the user-clustered bootstrap resampling unit, bootstrap replicate count, CI method, confidence level, handling of repeated observations within users, multiplicity policy, cross-seed aggregation rule, seed-stability rule, `delta`, and the procedure used to choose `delta`. `delta` may not be selected or revised after inspecting M1 test outcomes.

```text
no decisive winner
```

A meaningful pairwise reversal requires decisive preference for Model A over Model B in one condition and decisive preference for Model B over Model A in another. Tiny sign flips that fall within uncertainty or below the practical-effect threshold are not counted as meaningful reversals.

Define **decisive pairwise reversal**:

```text
R_decisive(A, B; c1, c2) = 1
iff A is decisively preferred to B under c1
and B is decisively preferred to A under c2
else 0
```

A raw sign change that does not meet the decisive-winner rule may be recorded separately as a `nominal_order_flip`, but it is not a meaningful reversal.

### Direct pairwise-margin contrast estimands

For each core model pair `A, B` and evaluation condition `c`, define the pairwise margin:

```text
D_AB(c) = Metric_A(c) - Metric_B(c)
```

For a fixed behavioral target `t`, define the logging-regime contrast:

```text
G_AB(t) = D_AB(standard, t) - D_AB(randomized, t)
```

For a fixed logging regime `r`, define the behavioral-target contrast:

```text
T_AB(r) = D_AB(r, target_A) - D_AB(r, target_B)
```

Report the point estimate and confidence interval for every primary `G_AB` and `T_AB`. These contrasts directly estimate whether the model-comparison margin changes across logging regimes or behavioral targets; they complement, rather than replace, the decisive-winner and reversal classifications.

M0 must freeze the clustered resampling procedure for these contrasts before M1. For matched-support analyses with the same eligible users in both compared conditions, use paired user-level resampling. For native-regime contrasts whose evaluated populations are not identical, use a predeclared user-clustered contrast procedure appropriate to the verified overlap structure; do not falsely treat non-identical native populations as paired.

### Reversal taxonomy

Classify primary stability results using the following predeclared labels:

```text
R0 — Stable
No decisive ordering change.

R1 — Nominal flip
Raw ordering changes, but the change is not decisive under the
frozen uncertainty + practical-effect rule.

R2 — Composition-sensitive reversal
A decisive reversal appears in native-regime evaluation but
disappears under the required matched-support analysis.

R3 — Robust regime reversal
A decisive logging-regime reversal persists under matched support
and the prespecified robustness checks.

R4 — Target reversal
A decisive model ordering change occurs between the two frozen
behavioral targets.
```

These labels are descriptive classifications, not a single aggregate stability score. If more than one label applies, report each applicable classification rather than forcing one mutually exclusive category.

Report both:

- **winner stability**
- **pairwise reversal structure**

Do not collapse all exploratory conditions into one headline “stability score.”

## 6.2 Primary decision depth

Choose exactly one primary K in M0.

Example:

```text
K = 10
```

Other K values are sensitivity analyses only.

## 6.3 Primary endpoint

Use **one common primary ranking-selection metric across both behavioral targets** so that logging regime and target are the intended moving parts while the evaluation rule remains fixed.

Presumptive M0 choice:

```text
Primary K: 10
Primary metric: NDCG@10
```

M0 may replace this common metric only if it establishes a concrete semantic reason that the same metric is invalid for one of the two frozen targets. Any replacement must be justified and frozen before comparative test evaluation; it must not be chosen based on which metric produces the most interesting reversal.

Secondary metrics can include:

- Recall@K
- HitRate@K
- exposure concentration
- coverage
- diversity

Catalog-distribution metrics are secondary unless the primary study reveals a reason to elevate them.

## 6.4 Practical effect reporting

For every core comparison report:

- absolute difference
- relative difference where interpretable
- confidence interval
- practical magnitude relative to a frozen reference or threshold if one is defined

Do not equate statistical detectability with substantive importance.

## 6.5 Uncertainty decomposition

Required for M1:

### Evaluation uncertainty

- paired **user-clustered bootstrap** for all core pairwise differences
- resample users, preserving each sampled user's repeated exposures within the replicate
- aggregate pairwise differences across the three frozen training seeds using the frozen cross-seed rule
- confidence intervals for core pairwise differences
- confidence intervals for the frozen logging-regime contrasts `G_AB` and behavioral-target contrasts `T_AB` using the predeclared clustered procedure appropriate to the compared populations

### Training uncertainty

For stochastic models:

- exactly 3 frozen seeds for M1
- report per-seed primary metrics plus mean and standard deviation across seeds
- report whether each relevant pairwise ordering is training-seed-stable under the frozen rule
- a candidate winner cannot be declared decisive if any required pair is training-seed-unstable

### Required matched-support companion analysis

Every standard-vs-randomized primary comparison must report both:

```text
A. native-regime estimate
B. matched-support estimate
```

### Frozen matched-support policy

M0 must define the matched-support population using descriptive regime information only, before inspecting comparative model results. Use this presumptive restriction hierarchy unless KuaiRand semantics make a step invalid:

1. shared calendar window
2. shared eligible users
3. compatible item support
4. identical target availability
5. identical applicable UI / scenario restrictions

The final restriction rule is frozen before M1 model comparisons. No alternative matching rule may be promoted to primary status because it strengthens or weakens a reversal.

For both native and matched-support analyses, report:

- number of users
- number of items
- number of exposures / ranking units
- retained fraction after restriction
- target prevalence before and after restriction

This analysis is required because an apparent reversal may disappear after population-composition differences are controlled through restriction. Such a result is itself a primary finding, not a failed sensitivity check.

### Protocol sensitivity

Keep additional sensitivity analysis modest initially.

Run only a small number of prespecified checks such as:

- reasonable target-threshold alternative if applicable
- scenario restriction if target semantics differ by UI context

Do not build a large inferential framework before the primary result exists.

---

# 7. User and item cohorts

Cohorts must be defined from **pre-existing behavior**, not from outcome differences observed after evaluation.

## User history

Use precise terminology:

- zero-shot user — no collaborative history; only include if the protocol can support a meaningful evaluation
- few-shot / low-history user
- medium-history user
- high-history user

Do not call low-history users “cold start” unless they truly have no collaborative history.

## Item history

Likewise distinguish:

- zero-shot item
- few-shot / low-history item
- established item

## Interest entropy

- narrow
- mixed
- broad

## Mainstream affinity

- head-heavy
- balanced
- long-tail

## Eligibility audit

The benchmark report must show:

- fraction of original users retained
- fraction excluded by history requirements
- activity distribution before and after filtering
- whether eligibility rules disproportionately retain heavy users

This guards against survivorship bias in cohort conclusions.

---

# 8. Explanatory features [Conditional]

This section is **inactive unless M1 establishes a real model-selection reversal or instability worth explaining**.

The purpose of features is explanatory:

> **Which observed user/item characteristics account for the reversal?**

## 8.1 Keep feature families conceptually separate

### External / provided item information

Examples:

- category/content attributes
- duration
- static item descriptors

### Provided user attributes

Use only if their provenance and appropriateness are clear.

Potentially sensitive or poorly justified attributes should be excluded even if predictive.

### History-derived behavioral features

Examples:

- user history length
- recent engagement rate
- item exposure count
- item historical engagement rate
- interaction entropy
- category preference distribution

Do not call history-derived summaries “side information.”

## 8.2 Feature-source provenance

For every feature family record whether it is constructed from:

```text
standard-policy history only
randomized history
both
```

The core explanatory analysis should prefer standard-history-derived features so randomized evaluation remains a held-out logging regime.

## 8.3 Explicit popularity/exposure ablation

A comparison such as:

```text
hybrid
vs
hybrid minus explicit popularity/exposure feature
```

is an **ablation**, not a causal control.

Popularity/exposure may remain encoded indirectly through representation density and correlated metadata.

## 8.4 Continuous-history analysis

If instability depends on history, analyze it continuously:

```text
user history length -> reversal magnitude / model difference
item history count  -> reversal magnitude / model difference
```

Buckets may be shown for readability but should not be the only analysis.

---


## 8.5 Frozen M0 protocol artifact

M0 must emit one authoritative machine-readable protocol artifact:

```text
research/protocol_frozen_m0.yaml
```

The artifact must contain the machine-relevant frozen decisions or references to the canonical config/target files that contain them, including at minimum:

- dataset version/checksum and temporal windows
- training interaction definition, objective, negative policy, and frozen seed set
- exact Model A/B/C identities and hyperparameter-search budgets
- two frozen target definitions
- ranking unit and candidate-set rules
- exact primary-metric and NDCG edge-case semantics
- primary K, aggregation unit, and weighting policy
- decisive-winner rule, `delta`, confidence level, bootstrap settings, multiplicity policy, cross-seed aggregation, and seed-stability rule
- direct `D_AB`, `G_AB`, and `T_AB` estimands and their clustered uncertainty procedures
- matched-support rule
- invalidation criteria

Compute a SHA-256 hash over the canonical bytes of `research/protocol_frozen_m0.yaml` **without embedding `protocol_sha256` inside that hashed payload**. Record the resulting hash externally in primary M1 experiment manifests (or an adjacent metadata file) as `protocol_sha256`. Every primary M1 experiment manifest must reference that exact hash.

> **An M1 result that does not reference the frozen M0 `protocol_sha256` is not part of the valid primary study.**

Any change to a frozen primary decision after M0 requires a new protocol artifact/hash and must be documented as a protocol revision; results from different protocol hashes must not be silently pooled into one primary analysis.

# 9. Milestone roadmap

Only M0 and M1 are guaranteed.

M2 and M3 are conditional.

## M0 — Benchmark Integrity

### Goal

Build a reproducible primary study around:

```text
exactly 3 primary models
x
2 behavioral targets
x
2 logging regimes
x
1 common primary metric
x
1 primary K
```

### Required work

- verify KuaiRand randomized-exposure semantics
- verify the default targets `is_click` and `long_view`
- create target-definition cards
- audit standard vs randomized regime comparability
- freeze standard-only training protocol and primary training supervision
- freeze positive-interaction definition, non-positive treatment, loss/objective, validation objective, search spaces, budgets, and seed policy
- freeze the exact Model A/B/C set: popularity/engagement, BPR MF, LightGCN
- freeze temporal windows
- freeze logged-candidate evaluation unit, candidate construction, aggregation, weighting, `< K` handling, zero-relevance handling, score-tie policy, and exact NDCG gain/discount/relevance semantics
- implement deterministic hand-calculated metric-fixture tests
- freeze one common primary metric across both targets
- choose primary K
- define model-selection rule
- freeze the R0–R4 reversal taxonomy
- define executable decisive-winner / no-decision algorithm
- freeze user-clustered bootstrap unit, replicate count, CI method, confidence level, multiplicity policy, repeated-observation handling, cross-seed aggregation, and seed-stability rule
- freeze minimum practical-effect threshold and its pre-test selection procedure
- define practical effect reporting
- implement seed-aggregated paired user-clustered bootstrap intervals and 3-seed training-stability checks
- freeze native-regime and matched-support companion analyses, including the support-restriction hierarchy and retention reporting
- freeze the direct pairwise-margin estimands `D_AB`, logging-regime contrasts `G_AB`, behavioral-target contrasts `T_AB`, and their clustered uncertainty procedures
- emit `research/protocol_frozen_m0.yaml`, compute `protocol_sha256`, and require that hash in every primary M1 manifest
- define feature-source provenance rules
- create invalidation criteria

### Baselines / candidate models

Freeze exactly these three primary systems for M1:

- **Model A — Popularity / engagement baseline**
- **Model B — BPR matrix factorization**
- **Model C — LightGCN**

The two personalized models must be tuned competently before conclusions are drawn. BPR and LightGCN must receive comparable, predeclared hyperparameter-search budgets and must use validation data from the frozen pre-test period only. Hyperparameter spaces, trial budgets, early-stopping rules, and selection criteria are frozen in M0.

The purpose of the third model is to ensure that any important reversal persists beyond a trivial popularity-vs-personalization comparison. Model choice is now frozen; alternative architectures are not substituted after comparative test results are observed.

### Exit criterion

M1 does not begin until all of the following are documented and frozen:

- regime comparability and target semantics
- primary training supervision, exact checkpoints per frozen seed, cross-seed aggregation, and seed-stability policy
- logged-evaluation unit and candidate construction
- primary metric, K, and decisive-winner procedure
- matched-support restriction rule
- hyperparameter-search spaces and comparable search budgets
- exact NDCG edge-case semantics and passing metric-fixture tests
- direct contrast estimands and their clustered uncertainty procedures
- frozen M0 protocol artifact and `protocol_sha256`

---

## M1 — Primary Model-Selection Stability Study

### Research question

> **How stable is the offline model-selection decision across logging regimes and behavioral targets when the same three trained systems are held fixed?**

### Training

```text
standard-policy history only
one frozen interaction definition
one fitted checkpoint per model/frozen seed
no target-specific retraining or retuning
```

For each frozen seed, the identical fitted checkpoint is reused across all four regime × target evaluation conditions.

### Evaluation conditions

```text
standard exposure x target A
randomized exposure x target A
standard exposure x target B
randomized exposure x target B
```

### Primary result package

M1 must emit four linked artifacts.

#### 1. Selection table

| Evaluation regime | Target | Model A | Model B | Model C | Selected |
|---|---|---:|---:|---:|---|
| Standard exposure | Target A | ... | ... | ... | ... |
| Randomized exposure | Target A | ... | ... | ... | ... |
| Standard exposure | Target B | ... | ... | ... | ... |
| Randomized exposure | Target B | ... | ... | ... | ... |

`Selected` must permit `no decisive winner`.

#### 2. Pairwise-difference table

For every core model pair, report:

```text
paired difference
confidence interval
minimum practical-effect threshold
decisive / inconclusive
```

#### 3. Contrast table

For every core model pair, report the direct change in pairwise margin:

```text
D_AB(condition)
G_AB(target): standard-vs-randomized margin change + CI
T_AB(regime): target-A-vs-target-B margin change + CI
```

The contrast table must state whether each comparison used paired matched-support resampling or the frozen native-regime clustered contrast procedure.

#### 4. Stability table

For every regime × target comparison, report:

```text
native-regime winner
matched-support winner
decisive / inconclusive
reversal classification
```

The result package, not a single score table, is the center of gravity of M1.

### Required interpretation

For every apparent reversal, interpret the result in this order:

1. **Decisiveness** — Is the pairwise difference decisive under the frozen uncertainty + practical-effect rule? If not, classify it as `R1 — Nominal flip` rather than a meaningful reversal.
2. **Training uncertainty** — For stochastic models, is the relevant ordering stable across the prespecified training seeds?
3. **Matched support** — Does the reversal persist in the required matched-support companion analysis? If a decisive native-regime reversal disappears after matching, classify it as `R2 — Composition-sensitive reversal`.
4. **Protocol sensitivity** — Does the finding survive the small set of prespecified protocol checks?
5. **Residual composition** — Could remaining differences in regime composition plausibly explain the result?
6. **Robust classification** — Only after the preceding checks may a logging-regime change be classified as `R3 — Robust regime reversal`. A decisive ordering change across the two frozen targets is reported as `R4 — Target reversal`.
7. **Explanation** — Only after a practically meaningful phenomenon survives this sequence should RankLab activate explanatory M2 analysis.

### Exit paths

#### No meaningful reversal

That is a valid result.

RankLab may stop or perform one prespecified robustness check.

#### Meaningful reversal

Proceed to M2 to explain it.

---

## M2 — Explain the Reversal [Conditional]

### Entry requirement

At least one practically meaningful, uncertainty-robust model-selection reversal from M1.

### Research question

> **What explains the instability?**

Candidate explanatory axes:

- user history
- item history
- item exposure frequency
- target prevalence
- regime composition
- external item metadata
- provided user attributes
- history-derived representations

### Rules

- matched-support analyses come before feature-heavy modeling
- side features are explanatory tools, not a predetermined product
- use continuous history analyses where possible
- avoid causal language unless the design supports it

### Deliverable

A decomposition of the reversal into plausible contributing factors.

---

## M3 — Intervention [Conditional]

### Entry requirement

M2 identifies a concrete harmful tradeoff or instability mechanism that can plausibly be changed.

Possible examples:

- excessive exposure concentration
- a specific subgroup instability
- a target-specific ranking conflict

### Research question

> **Can a targeted intervention improve the identified problem without creating a larger loss elsewhere?**

Possible interventions are selected from the M2 finding, not predeclared for portfolio breadth.

Reranking is one option, not a guaranteed milestone.

---

## M4 — Evidence-Driven Extension

Only if M3 or replication genuinely requires it.

Possible directions:

- training on mixed standard + randomized logs
- external replication
- candidate-generation study
- a new feature family

Explicitly excluded by default:

- contextual bandits
- reinforcement learning
- simulated long-term user behavior
- architecture tourism

---

# 10. Decision-language discipline

Allowed:

> Under randomized exposure, Model B ranks above Model A for Target A.

> The selected model changes between logging regimes.

> The reversal persists after restricting to shared users and item support.

> Removing the explicit exposure-frequency feature reduces the observed difference.

Avoid:

> Random exposure reveals the true best model.

> Standard logs are biased and random logs are unbiased truth.

> Model B is better for users.

> Feature X caused the reversal.

The project studies **selection stability under different logged conditions**, not a privileged ground-truth universe.

---

# 11. Experiment validity rules

A comparison is invalid if any of the following differ unexpectedly between compared runs:

- dataset version
- dataset checksum
- temporal split
- user population
- test population
- candidate pool
- candidate-generation depth
- negative-sampling policy
- feature cutoff policy
- metric implementation
- test-time feature availability
- test leakage
- hyperparameter selection using test results
- model-selection rule
- random-seed policy where determinism is required

Invalid runs should remain in the repository under an explicit location such as:

```text
research/
  valid/
  invalidated/
```

Each invalidated experiment should record:

- experiment ID
- reason for invalidation
- discovery date
- affected claims
- whether a rerun supersedes it

---

# 12. Recommended repository structure

Start lean:

```text
ranklab/
│
├── README.md
├── pyproject.toml
├── LICENSE
│
├── configs/
│   ├── benchmark.yaml
│   ├── models/
│   └── experiments/
│
├── data/
│   └── README.md
│
├── research/
│   ├── protocol.md
│   ├── protocol_frozen_m0.yaml
│   ├── targets/
│   ├── valid/
│   └── invalidated/
│
├── src/ranklab/
│   ├── data/
│   ├── benchmark/
│   ├── models/
│   ├── evaluation/
│   ├── analysis/
│   └── artifacts/
│
├── tests/
│
└── reports/
    ├── figures/
    ├── tables/
    └── results/
```

Add `features/`, `cohorts/`, or `reranking/` only if M2 or M3 actually requires them.

---

# 13. CLI target

The initial CLI should match the actual primary study.

Example:

```bash
ranklab evaluate   --benchmark configs/benchmark.yaml   --model configs/models/mf.yaml   --eval-regime randomized   --target long_view
```

Do not expose feature or reranking flags before those capabilities exist.

---

# 14. README target structure

The README should lead with the empirical result, not methodological machinery.

Recommended order:

1. Project title and one-sentence research question
2. Hero result: the 2×2 logging-regime × target comparison
3. Key finding or null result
4. Why it matters
5. Experimental design
6. Standard-policy vs randomized exposure
7. Target A vs Target B
8. Why selection changed, only if M2 establishes an explanation
9. Robustness checks
10. Reproduction instructions
11. Limitations
12. Invalidated experiments / research history

Do **not** promise:

- Pareto charts
- failure taxonomies
- side-feature findings
- reranking results

unless the project actually reaches those stages.

---

# 15. Explicit non-goals

RankLab should **not** drift into:

- a Netflix UI clone
- a short-video recommendation frontend
- an LLM recommendation system
- a showcase of every KuaiRand signal
- using all available user/item features
- dozens of recommender architectures
- claiming randomized exposure is unbiased ground truth
- full-catalog ranking as a default primary evaluation
- pretending logged exposures are explicit impression slates without verification
- contextual bandits or RL by default
- industrial-scale serving claims
- causal claims from simple feature ablations
- declaring statistically significant tiny effects “important”
- treating every target/metric/K combination as confirmatory
- adding reranking without an M2 mechanism to address
- claiming a universally best recommender

---

# 16. Scope-control rule

Any proposed new feature, model, dataset, metric, cohort, or experiment must answer at least one of:

1. Which frozen hypothesis does it test?
2. Which existing ambiguity does it resolve?
3. Which observed failure mode does it isolate?
4. Which primary conclusion could it falsify?
5. Which benchmark sensitivity does it test?

If it answers none of these, it does not enter the project.

New work should be rejected if its main justification is:

- “industry uses it”
- “it looks production-like”
- “it would be good for the portfolio”
- “it is a modern architecture”
- “it would make the roadmap more complete”

---

# 17. Milestone summary

```text
M0  Benchmark Integrity
    - verify logging regimes
    - validate/reject 2 targets
    - fixed models: popularity, BPR MF, LightGCN
    - frozen training supervision + search budgets
    - frozen logged-evaluation unit + exact NDCG semantics
    - deterministic metric-fixture tests
    - 1 common primary metric
    - 1 primary K
    - standard-only training
    - comparability audit
    - executable decisive-winner algorithm
    - frozen clustered-bootstrap procedure + practical threshold
    - R0–R4 reversal taxonomy
    - frozen native + matched-support policy
    - direct D/G/T contrast estimands + clustered uncertainty
    - frozen M0 protocol artifact + protocol_sha256
    - uncertainty decomposition

M1  Primary Stability Study
    same 3 trained models
    x 2 logging regimes
    x 2 behavioral targets
    x native + matched-support interpretation
    + direct regime/target margin contrasts

M2  Explain the reversal [conditional]

M3  Targeted intervention [conditional]

M4  Evidence-driven extension [optional]
```

The project may legitimately end after M1.

That is acceptable if the null result is rigorous and informative.

---

# 18. Final target narrative

RankLab should ultimately support a claim no stronger than its evidence.

A strong target narrative is:

> **RankLab tests how stable offline recommender model selection is when the same systems are evaluated under standard-policy versus randomized exposure and across two predeclared behavioral targets, initially `is_click` and `long_view`.**

A strong positive finding would look like:

> **The preferred model changed under randomized exposure for one behavioral target, and the reversal persisted after matching user/item support and accounting for evaluation uncertainty.**

A strong explanatory finding would look like:

> **The reversal was concentrated among low-history users and high-exposure items, while remaining small elsewhere.**

A strong null finding would also be acceptable:

> **Model selection remained stable across both logging regimes and both targets, and apparent reversals were small relative to uncertainty and regime-composition effects.**

Only after such a primary result should RankLab introduce side-feature analysis or an intervention.

Most importantly:

> **RankLab is not a KuaiRand feature showcase. It is a controlled study of whether the data-collection regime and behavioral target change the model an offline evaluation tells us to select.**

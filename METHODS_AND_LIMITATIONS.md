# FairLens — Methods & Limitations

*A write-up section for the FYP report / paper. Describes what the FairLens
pipeline does, how it is evaluated, why its Adult Census results differ from
Loganathan et al. (2025), and the known limitations of the implementation.*

---

## 1. Methodology

FairLens is a faithful re-implementation of the three-stage bias-mitigation
framework of Loganathan et al. (2025) — **Reweighing → Adversarial Debiasing
(ADB) → Calibrated Equalised Odds (CEO)** — evaluated across four models
(Random Forest, XGBoost, LightGBM, TabNet) and several datasets (Adult Census,
COMPAS, German Credit, Taiwan Credit), with an exploratory image-domain
extension (Fitzpatrick17k). All fairness computation uses IBM's AIF360.

**Data preparation.** Missing values (`?`) are mode-imputed rather than dropped,
retaining the full 48,842 Adult Census rows. Categorical features are
label-encoded (matching AIF360's `AdultDataset` convention). The target and the
protected attribute are binarised; the privileged group is mapped to 1.

**Splitting.** The data is split, stratified on the target, into
60 % train / 10 % validation / 30 % test. The validation split lets CEO be
calibrated without touching the test set.

**Leakage control.** Every fitted transform is estimated on the training split
only: the `StandardScaler` (for the ADB network), the Reweighing weights, and all
SMOTE resampling. The test set is never resampled or used to fit any transform.
**CEO's cost-constraint (`fnr` / `fpr` / `weighted`) is selected on the
validation set and only then evaluated on the test set** — model selection never
sees the test data.

**Reproducibility & variance.** The split, models, SMOTE, ADB (TensorFlow) and
CEO are all seeded. Metrics are reported as the **mean ± standard deviation over
five seeds (42–46)** so that run-to-run instability is visible.

**Robustness guards.**
- *Do-no-harm:* post-processing (CEO) is skipped and the Reweighing result kept
  when the smallest validation group is < 50, or when CEO produces a degenerate
  model. This prevents the model-collapse observed on small datasets.
- *Degenerate-model flag:* any stage whose accuracy falls below the
  majority-class baseline is flagged; its fairness metrics are not treated as
  meaningful (this catches "fair-looking but broken" models, e.g. from SMOTE).

**Metrics.** Fairness: Statistical Parity Difference (SPD), Disparate Impact
(DI), Average Odds Difference (AOD), Equal-Opportunity Difference (EOD, the TPR
gap). Performance: Accuracy and Balanced Accuracy. Fair ranges: |SPD| < 0.05,
0.8 ≤ DI ≤ 1.25, |AOD| < 0.1, |EOD| < 0.1.

**Dataset-level vs prediction-level disparity.** The "baseline" SPD/DI shown
before training measure disparity in the *dataset's true label base rates*
(`metric_basis = dataset_base_rate_true_labels`); these are reported separately
from each model's `original`-stage SPD/DI, which measure *prediction* disparity
on the held-out test set.

**SMOTE study.** Each SMOTE variant is fit on the training split, and every
selected model is retrained on the resampled data and evaluated on the common
test set, producing a full model × variant matrix.

---

## 2. Key methodological findings

**(a) The ADB and CEO stages are model-agnostic.** AIF360's Adversarial
Debiasing is a standalone classifier that does not wrap the base estimator. Since
the Reweighing weights depend only on the protected attribute and label, ADB's
input — and therefore its output, and CEO's output calibrated on it — is
identical for every model. **Only the Baseline and Reweighing stages are
model-specific.** In an early unseeded run the ADB stage appeared model-specific
(DI ranging 0.68–0.96); seeding revealed these differences to be
random-initialisation noise.

**(b) ADB is highly unstable.** Across five seeds, the ADB-stage DI has a
standard deviation of ≈ 0.22 (mean ≈ 0.70 on Adult) — DI varies from ≈ 0.46 to
≈ 0.94 — whereas the Reweighing stage is stable (DI std ≈ 0.03). A single ADB/CEO
run is therefore not reliable; Reweighing gives a smaller but far steadier
improvement.

---

## 3. Why our Adult Census results differ from the paper

**(a) Higher baseline accuracy (≈0.86 vs the paper's ≈0.80).** Expected — we
retain all 14 features and impute missing rows, whereas the paper's AIF360
`AdultDataset` drops rows and columns. Different preprocessing, not a different
method.

**(b) We could not reproduce the reported ≈0.96 post-mitigation accuracy.**
Under strict, leakage-controlled evaluation the full pipeline settles at
**≈0.84 accuracy**, reducing bias from SPD −0.101 / DI 0.60 (dataset) to
**SPD −0.035 ± 0.038 / DI 0.796 ± 0.223** (final, identical across models). This
~0.84 figure is consistent with an independent published comparison of the same
AIF360/Fairlearn techniques on Adult (~0.78–0.85 across all methods).

We do **not** assert a specific error in the original work, whose code we cannot
inspect. We report only that its accuracy was not reproducible under our
protocol, most plausibly due to differences in evaluation-set handling
(information leakage). Note also that simultaneously satisfying *all four*
fairness metrics — as the paper reports — is only easy when the classifier is
near-perfect, which is consistent with an inflated-accuracy regime. Under honest
accuracy, CEO (an equalised-odds method) brings AOD/EOD within range while DI
reaches only its ≈ 0.80 boundary.

---

## 4. Limitations

1. **Model-agnostic in-processing.** Because AIF360 ADB does not use the base
   model (see §2a), the "four models × ADB/CEO" comparison collapses to a single
   shared classifier at those stages. This is a property of the toolkit, and is
   disclosed rather than hidden.

2. **ADB variance.** The high seed-to-seed variance of ADB (DI std ≈ 0.22) limits
   the precision of any single ADB/CEO result; we report five-seed means ± std.

3. **TabNet reweighing is approximate.** For the tree models, Reweighing uses
   scikit-learn's per-sample `sample_weight`. TabNet's `fit(weights=…)` instead
   drives a weighted *sampler*, so TabNet's reweighing is realised as weighted
   resampling rather than loss weighting.

4. **The protected attribute appears twice in the ADB feature matrix** — once as
   a (scaled) feature and once as the AIF360 protected attribute. This is
   redundant; it does not break debiasing but slightly complicates interpretation.

5. **Binarised protected attribute.** Race is treated as White vs non-white; we
   do not evaluate across all racial groups or intersectionally (race × sex).

6. **Categorical encoding.** Label encoding imposes an arbitrary ordering on
   nominal features.

7. **Small datasets.** ADB is a neural network and is unreliable below ~2,000
   rows; German Credit (n = 1,000) triggers the do-no-harm guard (CEO skipped,
   Reweighing retained), and the engine emits explicit small-dataset warnings.

8. **CEO uses group-conditional thresholds** (the protected attribute at decision
   time), raising the disparate-treatment consideration standard to
   post-processing fairness methods.

---

## 5. Summary of correctness measures

| Concern | How it is handled |
|---|---|
| Train/test leakage from resampling | SMOTE & Reweighing fit on train only; test never resampled |
| Scaler leakage | `StandardScaler` fit on train only |
| Post-processing leakage | CEO calibrated on validation; cost-constraint selected on validation, reported on test |
| Reproducibility | All random components seeded; results averaged over 5 seeds ± std |
| Run-to-run instability | Reported explicitly via standard deviation (exposes ADB variance) |
| Metric confusion | Dataset base-rate disparity reported separately from prediction disparity |
| Small-dataset collapse | Do-no-harm guard keeps Reweighing when CEO is unreliable |
| "Fair-looking but broken" models | Degenerate-model flag when accuracy < majority-class baseline |
| Cross-model generality | SMOTE effect measured across all selected models, not a single probe |

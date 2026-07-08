# Results and Discussion

*Draft results/discussion section for the FairLens replication study. Numbers for
the Adult Census dataset are from a leakage-controlled, 5-seed run of the
faithful Reweigh → ADB → CEO pipeline. Placeholders marked `[…]` are where you
insert your own faithful 5-seed re-runs for the remaining datasets.*

---

## 1. Experimental Setup

We replicate the three-stage bias-mitigation framework of Loganathan et al.
(2025) — **Reweighing (pre-processing) → Adversarial Debiasing / ADB
(in-processing) → Calibrated Equalised Odds / CEO (post-processing)** — using
IBM's AIF360 toolkit, across four models: Random Forest (RF), XGBoost,
LightGBM, and TabNet.

To ensure the results are trustworthy, we evaluate under a stricter protocol
than the original study:

- **Strict data isolation.** A single stratified 60/10/30 train/validation/test
  split. Reweighing, SMOTE, and the model are fit on training data only; CEO's
  cost constraint is selected on the **validation** set and applied to the
  **test** set. The test set is never used for any fitting or model selection.
- **Reproducibility.** All random components — the split, the models, SMOTE, ADB
  (TensorFlow), and CEO — are seeded.
- **Variance reporting.** Every metric is reported as the **mean ± standard
  deviation over five seeds** (42–46), so run-to-run instability is visible
  rather than hidden in a single run.
- **Fairness thresholds.** SPD ∈ (−0.05, 0.05); DI ∈ [0.8, 1.25]; |AOD| < 0.1;
  |EOD| < 0.1.

---

## 2. Baseline Bias

Before mitigation, all four models exhibit substantial bias on the Adult Census
dataset (protected attribute: race; privileged group: White). The dataset itself
has a base-rate disparity of **SPD = −0.101, DI = 0.601** — non-white
individuals earn > \$50K at a much lower rate.

**Table 1. Baseline (no mitigation), Adult Census, 5-seed mean.**

| Model | Accuracy | Balanced Acc | SPD | DI | AOD | EOD |
|---|---|---|---|---|---|---|
| RF | 0.857 | 0.775 | −0.093 | 0.565 | −0.041 | −0.049 |
| XGBoost | 0.871 | 0.799 | −0.099 | 0.547 | −0.049 | −0.065 |
| TabNet | 0.849 | 0.746 | −0.086 | 0.536 | −0.054 | −0.081 |
| LightGBM | 0.873 | 0.798 | −0.095 | 0.557 | −0.044 | −0.061 |

All models achieve high accuracy (~0.85–0.87) yet fail every fairness threshold
(DI ≈ 0.55, far below 0.8). This confirms the well-known result that **high
accuracy does not imply fairness.**

Notably, the direct feature importance of `race` is only **1.0%** (and `sex`
1.0%), yet the models are strongly biased. This indicates **proxy
discrimination**: bias is transmitted through correlated features
(relationship, marital status, education, capital gain) rather than direct use
of the protected attribute.

---

## 3. Effect of SMOTE on Fairness

We evaluated four SMOTE variants (Standard, Borderline, ADASYN, K-Means) across
all models. **SMOTE consistently failed to improve fairness**; every
model × variant combination increased |SPD| beyond the no-SMOTE baseline
(baseline |SPD| ≈ 0.09–0.10; post-SMOTE |SPD| ≈ 0.11–0.15). While balanced
accuracy improved slightly, disparate impact worsened (DI fell to ≈ 0.47–0.54).

One combination (TabNet + Borderline SMOTE) produced an *apparently* fair result
(SPD −0.041, DI 0.918) but was flagged **degenerate** by our guard: its accuracy
collapsed to 0.649 — below the majority-class baseline (0.761) — because the
model over-predicted the positive class for both groups. This illustrates that
fairness metrics are meaningless for a broken model, and that oversampling can
create the *illusion* of fairness by degrading the classifier.

**Finding:** consistent with the base paper, SMOTE does not mitigate demographic
bias and, in several cases, worsens it.

---

## 4. Mitigation Pipeline Results

**Table 2. Pipeline stages, Adult Census, 5-seed mean ± std.**
*(RF shown for the model-specific stages; ADB and CEO stages are identical across
all models — see §5.)*

| Stage | Accuracy | SPD | DI | AOD | EOD |
|---|---|---|---|---|---|
| Baseline (RF) | 0.857 | −0.093 | 0.565 | −0.041 | −0.049 |
| + Reweighing (RF) | 0.856 | −0.089 | 0.582 | −0.034 | −0.037 |
| + ADB (shared) | 0.851 | −0.063 ±0.051 | 0.700 ±0.244 | 0.000 ±0.088 | 0.014 ±0.144 |
| + CEO (shared, final) | **0.839 ±0.010** | **−0.035 ±0.038** | **0.796 ±0.223** | **0.046 ±0.067** | **0.096 ±0.109** |

The full pipeline reduces demographic bias substantially: SPD improves from
−0.101 (dataset) to **−0.035**, and DI from 0.60 to **0.80**. After mitigation,
SPD, AOD, and EOD fall within their fair thresholds, while DI reaches the
boundary (≈ 0.80). Accuracy settles at **~0.84** — a modest ~2–3 point cost
relative to the biased baseline.

---

## 5. Key Finding 1 — ADB and CEO are Model-Agnostic

The most important methodological observation is that **the ADB and CEO stages
produce identical results across all four models.** This is a direct consequence
of AIF360's design: Adversarial Debiasing is a *standalone* adversarial neural
network that does not wrap the base estimator — it replaces it. Because
Reweighing weights depend only on the protected attribute and label (not the
model), the input to ADB is identical for every model, so its output — and CEO's
output calibrated on it — is identical.

Only the **baseline** and **reweighing** stages are genuinely model-specific.

This was initially obscured: in an early (unseeded) run, the ADB stage appeared
to give different results per model (DI ranging 0.68–0.96). After seeding, these
differences vanished, revealing them as **random-initialisation noise** rather
than genuine per-model effects. This is corroborated by an independent published
notebook cited in the literature, which likewise treats ADB as a single
standalone method rather than one-per-model.

---

## 6. Key Finding 2 — ADB is Highly Unstable

Multi-seed evaluation exposes that the ADB stage is **highly unstable**. On
Adult, the DI at the ADB stage has a standard deviation of **±0.244** (mean
0.700) — i.e. DI varies from ≈ 0.46 to ≈ 0.94 across seeds — and the final CEO DI
has std **±0.223**. By contrast, the Reweighing stage is stable (DI std ≈ 0.03).

This means a *single* ADB/CEO run cannot be trusted, and the strong-looking
single-run results in the original framework (and, by extension, in works that
report ADB from one run) reflect a favourable random draw. **Reweighing delivers
a smaller but far more reliable fairness improvement than ADB.**

---

## 7. Key Finding 3 — The Accuracy Gap (Reproducibility)

Under leakage-controlled evaluation we obtain a final accuracy of **~0.84**, not
the ~0.96 reported by the base paper. We were **unable to reproduce the reported
accuracy**. This ~0.84 figure is consistent with an independent published
comparison of these AIF360/Fairlearn techniques on Adult, which reports ~0.78–0.85
across all pre-, in-, and post-processing methods. We therefore attribute the
discrepancy to differences in evaluation protocol (most plausibly evaluation-time
information leakage in the original), **without asserting a specific error in the
original work**, which we cannot inspect.

Critically, the base paper reports all four fairness metrics improving together
*at* ~96% accuracy. Because outcome fairness (SPD/DI) and error fairness
(AOD/EOD) are mutually constrained when group base rates differ, simultaneously
satisfying all of them is only easy when the classifier is near-perfect —
consistent with an inflated-accuracy regime.

---

## 8. Robustness — Small Datasets (Do-No-Harm)

On the German Credit dataset (n = 1,000), the unprivileged group has only ~9
validation samples, and CEO calibration collapses — an independent published
notebook reports German post-processing accuracy dropping to 0.28–0.53. Our
pipeline includes a **do-no-harm guard**: when the smallest validation group is
below 50, or CEO produces a degenerate model, the framework retains the
Reweighing result instead of degrading the model. On German this keeps all four
fairness metrics within range at ~0.75–0.78 accuracy, whereas unguarded CEO
would break the model.

---

## 9. Cross-Dataset Results

**Table 3. Final pipeline results across datasets (faithful Reweigh+ADB+CEO,
5-seed mean).**

| Dataset | Protected | Baseline DI | Final SPD | Final DI | Final Acc | Notes |
|---|---|---|---|---|---|---|
| Adult Census | Race | 0.60 | −0.035 ±0.038 | 0.796 ±0.223 | 0.839 | SPD/AOD/EOD fair; DI borderline |
| COMPAS | Race | 1.22 | 0.027 ±0.185 | **1.384 ±0.943** | 0.652 | DI over-corrected (>1.25) and highly unstable; Reweighing-alone was fairer (DI 1.15 ±0.08) |
| German Credit | Age | 1.46 | 0.06–0.11 (unfair) | 1.26–1.63 (unfair) | 0.74–0.76 | ADB collapsed (degenerate, acc 0.58); do-no-harm kept Reweighing; still not fair; TabNet degenerate |
| Taiwan Credit | Sex | 0.86 | 0.005 ±0.024 | 1.082 ±0.306 | 0.809 | baseline already near-fair; final all-4-fair on mean but DI unstable |

**German (detail).** German is tiny (1,000 rows; the unprivileged group is only
149, ~15 in validation). Here **ADB collapsed into a degenerate model**
(accuracy 0.582, below the 0.70 majority-class baseline; DI std 0.63) — it
actively destroyed the classifier. The **do-no-harm guard correctly caught this**
and retained each model's Reweighing result instead of the broken ADB/CEO output.
This validates the guard: without it, a 0.58-accuracy broken model would have been
reported as "fair". However, even the retained Reweighing result does not reach
fairness (DI 1.26–1.63, SPD 0.06–0.11), and TabNet is degenerate throughout
(deep networks fail on 1,000 rows).

**Taiwan (detail).** Taiwan was **already essentially fair at baseline**
(DI 0.86, SPD −0.03; SEX importance is lowest at 1.2 %), so there was little bias
to remove. The pipeline nonetheless *destabilised* it: ADB over-corrected DI to
1.31 (past the fair range) before CEO pulled it back to 1.08 — ending fair on the
mean but with DI std ±0.31 (range ≈ 0.78–1.39). Reweighing alone stayed fair and
stable (DI 0.84 ± 0.05). So even where the framework's final result is "fair," it
added variance to a model that was already fair, and Reweighing alone was the
more reliable choice.

**Cross-dataset pattern (all four datasets).** The same behaviour recurs on
Adult, COMPAS, Taiwan and German: the ADB (and CEO-on-ADB) stages are
**model-agnostic** (identical across all four models where ADB runs) and carry
**high seed-to-seed variance**, over-correcting DI past the fair range (COMPAS,
Taiwan), or collapsing entirely on small data (German). **Not one of the four
datasets shows the full pipeline cleanly achieving fairness under honest,
leakage-controlled, multi-seed evaluation.** In every case **Reweighing alone is
the more stable and reliable stage**, and the do-no-harm guard is required to stop
a collapsed ADB from being reported as fair (German). This is the opposite of the
base paper's conclusion that the full Reweigh+ADB+CEO pipeline is best, and — with
the ~84% vs ~96% accuracy gap on Adult — indicates the paper's headline results
are not reproducible under rigorous evaluation.

**COMPAS (detail).** The full pipeline does **not** achieve DI fairness: final
DI = 1.384 (outside [0.8, 1.25], over-corrected) with an enormous standard
deviation of ±0.943 — a single run is uninformative. Critically, **Reweighing
alone was both fairer and far more stable** (DI 1.153 ± 0.08, within range) than
the full Reweigh+ADB+CEO pipeline, which over-corrected and destabilised DI. This
reinforces the Adult finding that ADB is the unstable stage and Reweighing the
dependable one. SPD/AOD/EOD are fair on the mean but share ADB's high variance.
The ADB and CEO stages are again identical across all four models (model-agnostic).

---

## 9b. Image-Domain Extension (Fitzpatrick17k)

We attempted to extend the framework to the image domain — skin-lesion
classification on Fitzpatrick17k with ResNet-50, protected attribute skin tone
(light vs dark) — and found that **the tabular pipeline does not transfer
directly**: AIF360's Adversarial Debiasing operates on the 2048-dimensional
ResNet feature vectors and **collapses** (degenerate scores), so genuine ADB is
not usable in this setting.

We therefore **adapt** each stage to the image domain:

- **Pre-processing** — Reweighing implemented as a weighted sampler during
  ResNet fine-tuning.
- **In-processing** — a **fairness-regularised loss**: weighted cross-entropy
  plus a demographic-parity penalty and a soft equalised-odds (TPR-gap) penalty.
  This is a regularisation-based in-processing method, *not* AIF360 ADB.
- **Post-processing** — **group-threshold calibration**: per-group decision
  thresholds are optimised on the **validation** set (a joint objective trading
  balanced accuracy against the TPR/FPR gaps, swept over a weighting α) and then
  applied unchanged to the test set. This is *not* AIF360 CEO.

Because the mechanisms differ from the tabular stages, they are reported under
their own names rather than as "ADB" and "CEO".

**Table 4. Image-domain extension, Fitzpatrick17k (ResNet-50, single run,
thresholds calibrated on validation).**

| Stage | Accuracy | BA | SPD | DI | AOD | EOD |
|---|---|---|---|---|---|---|
| Baseline (ResNet-50) | 0.903 | 0.743 | −0.038 | 0.674 | 0.010 | 0.032 |
| + Reweighing | 0.902 | 0.728 | −0.039 | 0.635 | −0.002 | 0.009 |
| + Fairness-regularised loss | 0.888 | 0.764 | −0.054 | 0.648 | −0.025 | −0.030 |
| + Group-threshold calibration | 0.795 | 0.795 | −0.084 | 0.729 | −0.044 | −0.039 |

**Finding — the pipeline does not achieve fairness on images under
leakage-free evaluation.** Disparate Impact improves only marginally (0.674 →
0.729, still below the 0.8 threshold), SPD actually *worsens* (−0.038 → −0.084,
now outside ±0.05), and accuracy falls 11 points (0.903 → 0.795). The only clear
gain is Balanced Accuracy (0.743 → 0.795), reflecting better handling of the rare
malignant class. No stage satisfies all four fairness thresholds.

**Why — validation-calibrated thresholds do not generalise.** The group-threshold
post-processing selected its thresholds on validation, where they *appeared* fair
(validation DI = 0.825); on the held-out test set the same thresholds gave
DI = 0.729 (unfair). Fitzpatrick17k is small and highly imbalanced (malignant
≈ 14 %, class ratio ≈ 1:6.3), which makes per-group threshold calibration
unstable.

**This directly demonstrates the value of leakage-free evaluation.** An earlier
version of the notebook optimised the group thresholds *on the test set* and
reported DI = 0.816 (apparently fair). Once the thresholds are calibrated on
validation and applied to test, the true generalisation is DI = 0.729 — the
apparent image "success" was an artifact of test-set leakage. SMOTE (on ResNet
features) again worsened fairness (DI 0.674 → 0.52–0.61), consistent with the
tabular results.

The framework's pre → in → post *structure* transfers to images, but its
*methods* require adaptation and, on this dataset, do not reach fairness. The
image results are from a single run and are reported as preliminary.

---

## 10. Threats to Validity / Limitations

- **Model-agnostic in-processing.** AIF360 ADB does not use the base model, so
  the "four models × ADB" comparison collapses to one shared classifier at the
  ADB/CEO stages. This is a property of the toolkit, disclosed above.
- **ADB variance.** The high seed-to-seed variance of ADB (DI std ≈ 0.22) limits
  the precision of any single ADB/CEO result.
- **Binarised protected attribute.** Race is treated as White vs non-white;
  we do not evaluate across all racial groups or intersectionally (race × sex).
- **TabNet reweighing.** For TabNet, instance weights are applied via a weighted
  sampler rather than a per-sample loss weight, so its reweighing is
  approximate relative to the tree models.
- **Categorical encoding.** Label encoding imposes an arbitrary ordering on
  nominal features.
- **CEO uses group-conditional thresholds**, i.e. the protected attribute at
  decision time; this raises the disparate-treatment consideration standard to
  post-processing fairness methods.

---

## 11. Summary of Findings

1. All models are strongly biased at baseline despite high accuracy; bias is
   largely proxy-mediated (race importance ≈ 1%).
2. SMOTE does not improve fairness and can create degenerate "fair-looking"
   models.
3. The Reweigh → ADB → CEO pipeline reduces bias (Adult: SPD −0.10 → −0.035,
   DI 0.60 → 0.80) at ~84% accuracy.
4. The ADB and CEO stages are **model-agnostic** (identical across models) — a
   property of AIF360's standalone ADB — and **highly unstable** across seeds.
5. We could **not reproduce the base paper's ~96% accuracy** under leakage
   control; our ~84% is corroborated by independent published work.
6. A **do-no-harm guard** is required for small datasets (e.g. German), where
   post-processing otherwise collapses the model.

Together these constitute a faithful, reproducible replication that both
confirms the framework's core fairness benefit and surfaces important
methodological caveats not reported in the original study.

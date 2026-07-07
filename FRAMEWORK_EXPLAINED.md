# FairLens — What the Framework Does, in Plain English

*A non-technical explanation of what our bias-mitigation framework does, what we
found while building it, and where it honestly stands. Written so anyone —
supervisor, examiner, or teammate — can follow it.*

---

## 1. What the framework is for

Machine-learning models trained on real data often treat some groups of people
worse than others — for example, predicting that non-white applicants are less
likely to earn a high income, even when that reflects historical bias in the
data rather than reality. **FairLens detects that bias and reduces it.**

It works in two parts:

1. **Detect** — measure how unfairly a model treats groups, using four standard
   fairness numbers (SPD, DI, AOD, EOD).
2. **Mitigate** — apply the base paper's three-step recipe and measure again.

We replicate the framework of **Loganathan et al. (2025)**: **Reweighing →
Adversarial Debiasing → Calibrated Equalised Odds**, tested on four models
(Random Forest, XGBoost, LightGBM, TabNet).

---

## 2. The three steps (pre → in → post)

- **Reweighing (pre-processing):** before training, give more weight to the
  under-represented group so the model sees them "fairly". *This step uses your
  chosen model.*
- **Adversarial Debiasing / ADB (in-processing):** train a special neural
  network that is penalised whenever it relies on the protected attribute.
  *Important: this is a self-contained model — it does not use your RF/XGBoost/etc.*
- **Calibrated Equalised Odds / CEO (post-processing):** after training, adjust
  the decision thresholds so the error rates are balanced across groups.

---

## 3. What we found (the honest story)

**The framework does reduce bias.** On the Adult dataset, it moves the models
from clearly biased (Disparate Impact ≈ 0.60) to roughly fair (≈ 0.80), at a
realistic accuracy of about 84%.

But building it carefully surfaced four things the original study did not report:

**Finding 1 — Two of the three steps ignore your model.**
Adversarial Debiasing trains its *own* classifier and throws your model away, and
CEO is calibrated on ADB's output. So no matter which of the four models you
pick, the ADB and CEO results come out **identical**. Only the Baseline and
Reweighing steps actually depend on the model. In the dashboard these two steps
are labelled **"shared"** for this reason.

**Finding 2 — The "per-model" differences were random noise.**
Our very first run *looked* like each model got a different ADB result. Once we
fixed the random seeds (so runs are repeatable), those differences disappeared —
they were just luck of the draw, not real effects.

**Finding 3 — Adversarial Debiasing is unstable.**
Run it five times and its Disparate Impact swings from about 0.46 to 0.94. That
is a huge amount of variation. Reweighing, by contrast, gives a smaller but very
steady improvement. So a single ADB run cannot be trusted.

**Finding 4 — We could not reproduce the paper's 96% accuracy.**
Under a strict, leak-free evaluation we get about **84%**, not 96%. An
independent published notebook using the same tools also reports ~80–85%. We do
**not** claim the paper is wrong — we only report that we could not reproduce its
accuracy under leak-free evaluation, most likely because of differences in how
the test data was handled.

---

## 4. Making it robust — the "do-no-harm" guard

On a small dataset (German Credit, 1,000 rows) the post-processing step can
break the model — an independent notebook shows its accuracy collapsing to as
low as 0.28. We added a **do-no-harm guard**: if a group is too small to
calibrate safely, or the post-processing produces a broken model, the framework
keeps the safe Reweighing result instead. This keeps small datasets fair without
wrecking accuracy.

---

## 5. A word on SMOTE

SMOTE is a popular technique for balancing class sizes, and people often assume
it improves fairness. **It does not.** Across all our tests it left fairness the
same or made it worse. In one case it produced a model that *looked* fair but was
actually broken (its accuracy dropped below just always guessing the majority
class) — our guard flags exactly this kind of false result.

---

## 6. Extending to images (dermatology)

We also tried the framework on medical images — skin-lesion photos
(Fitzpatrick17k), where the "groups" are lighter vs darker skin tones. The plain
result: **the exact tabular recipe does not carry over.** Adversarial Debiasing,
which works on spreadsheet-style features, breaks down on the thousands of
numbers a deep image model produces. So we kept the same three-step *idea*
(before / during / after training) but swapped in image-appropriate techniques —
a fairness-aware training loss, and adjusting the decision cut-off separately for
each skin-tone group (chosen on validation data, never the test data). This
reduces skin-tone bias in the image model. The finding itself — *the framework
needs adapting for images* — is part of the contribution. (These image results
are from a single run and are reported as preliminary.)

---

## 7. Where the framework honestly stands

- ✅ It **detects and reduces bias** across multiple datasets (Adult, COMPAS,
  German, Taiwan), with realistic accuracy.
- ✅ It is **rigorous**: leak-free evaluation, repeatable seeds, results reported
  as an average over five runs with their variability, and safety guards for
  small or broken models.
- ✅ It surfaces **honest findings** the original study missed: the in-processing
  step is model-agnostic and unstable, and the reported accuracy could not be
  reproduced.

This is a **faithful replication study**: we implement the framework exactly,
report what we see, and are careful not to over-claim.

---

## 8. One-paragraph summary (for a slide)

> FairLens faithfully replicates a three-stage bias-mitigation framework
> (Reweighing → Adversarial Debiasing → Calibrated Equalised Odds) across four
> models and several datasets. Under strict, leak-free, multi-seed evaluation it
> reduces demographic bias (Adult: DI 0.60 → 0.80) at realistic accuracy (~84%).
> We further find that the adversarial-debiasing and post-processing stages are
> model-agnostic and highly unstable, that SMOTE does not improve fairness, and
> that the original study's reported ~96% accuracy could not be reproduced —
> results consistent with independent published work.

# 🔍 FairLens

> A reproducible replication and analysis of a three-stage bias-mitigation framework for machine-learning models.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![AIF360](https://img.shields.io/badge/AIF360-0.6-6f42c1)](https://aif360.res.ibm.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FairLens is a faithful, reproducible re-implementation of the bias-mitigation
framework of **Loganathan et al. (2025)** — **Reweighing → Adversarial Debiasing
(ADB) → Calibrated Equalised Odds (CEO)** — built on IBM's **AIF360** toolkit and
evaluated across four models (Random Forest, XGBoost, LightGBM, TabNet) and
multiple real-world datasets. It provides both an interactive dashboard and a
rigorously-evaluated research pipeline, and it reports several methodological
findings not surfaced in the original study.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Results Summary](#results-summary)
- [Evaluation Protocol](#evaluation-protocol)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage — Dashboard](#usage--dashboard)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Overview

Algorithmic bias in ML systems causes real harm — from credit systems that
disadvantage certain groups to recidivism tools that over-flag minority
defendants. FairLens:

- **Detects** bias using four standard fairness metrics — Statistical Parity
  Difference (SPD), Disparate Impact (DI), Average Odds Difference (AOD),
  Equal-Opportunity Difference (EOD).
- **Mitigates** bias with the paper's three-stage pipeline (pre- / in- /
  post-processing).
- **Reports honestly** — under a strict, leakage-controlled, multi-seed
  evaluation protocol that exposes what the framework really does.

This is a **replication study**: we reproduce the framework faithfully and report
what we observe. Where our results differ from the original, we describe the
difference without asserting error in a work whose code we cannot inspect.

---

## Key Findings

1. **High accuracy ≠ fairness.** All four models reach ~85–87% accuracy at
   baseline yet fail every fairness threshold (DI ≈ 0.55 on Adult).
2. **Bias is proxy-mediated.** The direct feature importance of `race` is ~1%,
   yet the models are strongly biased — bias flows through correlated features.
3. **SMOTE does not improve fairness** and, in several cases, worsens it; one
   variant produced a *degenerate* model that only *looked* fair.
4. **The framework reduces bias.** On Adult, the full pipeline moves SPD from
   −0.101 to **−0.035** and DI from 0.60 to **0.80** at ~84% accuracy.
5. **ADB and CEO are model-agnostic.** AIF360's Adversarial Debiasing is a
   standalone classifier that ignores the base model, so the ADB and CEO stages
   are **identical across all four models by design**. Apparent per-model
   differences in an early unseeded run were random-initialisation noise.
6. **ADB is highly unstable** — DI at the ADB stage varies with a standard
   deviation of ≈ 0.22 across seeds, whereas Reweighing is stable (≈ 0.03).
7. **We could not reproduce the reported ~96% accuracy**; under leakage-controlled
   evaluation we obtain **~84%**, consistent with independent published work.
8. **A "do-no-harm" guard is required** for small datasets (e.g. German Credit),
   where post-processing otherwise collapses the model.

---

## Results Summary

### Adult Census — full pipeline (5-seed mean ± std)

| Stage | Accuracy | SPD | DI | AOD | EOD |
|---|---|---|---|---|---|
| Baseline (RF) | 0.857 | −0.093 | 0.565 | −0.041 | −0.049 |
| + Reweighing (RF) | 0.856 | −0.089 | 0.582 | −0.034 | −0.037 |
| + ADB *(shared)* | 0.851 | −0.063 ±0.051 | 0.700 ±0.244 | 0.000 | 0.014 |
| **+ CEO *(shared, final)*** | **0.839 ±0.010** | **−0.035 ±0.038** | **0.796 ±0.223** | **0.046** | **0.096** |

> *"shared"* = the ADB and CEO stages are model-agnostic and therefore identical
> for RF, XGBoost, LightGBM and TabNet. Only the Baseline and Reweighing stages
> are model-specific.

After mitigation, SPD, AOD and EOD fall within their fair thresholds and DI
reaches the ≈ 0.80 boundary, at a modest ~2–3 point accuracy cost. See
[RESULTS_AND_DISCUSSION.md](RESULTS_AND_DISCUSSION.md) for the full analysis.

---

## Evaluation Protocol

FairLens evaluates under a stricter protocol than a single train/test run:

- **Strict data isolation** — one stratified 60/10/30 train/validation/test
  split; Reweighing, SMOTE and the model fit on training data only; CEO's cost
  constraint is selected on **validation** and applied to **test**. The test set
  is never used for fitting or selection.
- **Reproducibility** — the split, models, SMOTE, ADB (TensorFlow) and CEO are
  all seeded.
- **Variance reporting** — metrics are reported as **mean ± std over 5 seeds**.
- **Do-no-harm guard** — post-processing is skipped (Reweighing retained) when a
  validation group is too small (< 50) or CEO produces a degenerate model.
- **Degenerate-model guard** — results are flagged when accuracy falls below the
  majority-class baseline, so a broken model is never reported as "fair".

---

## Project Structure

```
FairLens/
│
├── backend/                        # FastAPI backend
│   ├── main.py                     # API routes, caching, multi-seed orchestration
│   ├── fairness_engine.py          # Core pipeline: Reweigh → ADB → CEO + guards
│   ├── _run_adb_subprocess.py      # ADB isolated in a seeded subprocess
│   └── requirements.txt
│
├── frontend/                       # React dashboard
│   └── src/components/             # Upload / Configure / Models / Training / Results
│
├── notebook/
│   └── Image_Dataset.ipynb         # Image-domain extension (Fitzpatrick17k, ResNet-50)
│
├── RESULTS_AND_DISCUSSION.md       # Full results + analysis
├── METHODS_AND_LIMITATIONS.md      # Methodology + limitations
├── FRAMEWORK_EXPLAINED.md          # Plain-language explanation
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+ · Node.js 18+ · pip

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

---

## Usage — Dashboard

1. **Upload** a CSV. Column types and candidate protected attributes are
   auto-detected.
2. **Configure** the target column, protected attribute, privileged value, and
   feature columns.
3. **Select models & mitigation** (RF / XGBoost / LightGBM / TabNet;
   Reweighing → ADB → CEO), SMOTE variants to test, test-split size, and the
   number of seeds (1 for a quick run, 5 for report-grade variance).
4. **Run** — the backend computes baseline metrics, the SMOTE study, and the
   full pipeline per model, averaged over seeds.
5. **View & export** — interactive tables and charts (SPD/DI/AOD/EOD per stage,
   feature importance, per-group confusion matrices), plus JSON and printable
   PDF report export.

---

## Datasets

| Dataset | Rows | Protected | Privileged | Domain |
|---|---|---|---|---|
| [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult) | 48,842 | Race | White | Income prediction |
| [COMPAS](https://github.com/propublica/compas-analysis) | 7,214 | Race | Caucasian | Recidivism prediction |
| [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) | 1,000 | Age (binary) | Older | Credit risk |
| [Taiwan Credit](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) | 30,000 | Sex | Male | Credit default |
| [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) | 16,574 | Skin tone | FST 1–2 | Dermatology (image extension) |

> ⚠️ Datasets are **not included** in this repository due to size and licensing.
> The image-domain extension (Fitzpatrick17k + ResNet-50) is provided as a
> Jupyter notebook and is exploratory.

---

## Pipeline

```
Input Dataset
      │
      ▼   Baseline evaluation (RF · XGBoost · LightGBM · TabNet)   ── model-specific
      │
      ▼   SMOTE study (Standard · Borderline · ADASYN · K-Means)   ── does oversampling help?
      │
      ▼   Reweighing  (Kamiran & Calders instance weights)         ── pre-processing, model-specific
      │
      ▼   Adversarial Debiasing (AIF360)                           ── in-processing, MODEL-AGNOSTIC
      │
      ▼   Calibrated Equalised Odds (validation-calibrated)        ── post-processing, MODEL-AGNOSTIC
      │
   Final metrics · SPD · DI · AOD · EOD   (+ do-no-harm guard)
```

**Fair ranges:** |SPD| < 0.05 · DI ∈ [0.8, 1.25] · |AOD| < 0.1 · |EOD| < 0.1

---

## Tech Stack

**Backend:** FastAPI · scikit-learn · XGBoost · LightGBM · PyTorch-TabNet ·
AIF360 (Reweighing, Adversarial Debiasing, Calibrated Equalised Odds) ·
TensorFlow (ADB) · imbalanced-learn (SMOTE)

**Frontend:** React 18 · Recharts · Tailwind CSS

**Image extension:** PyTorch + torchvision (ResNet-50)

---

## Documentation

- [RESULTS_AND_DISCUSSION.md](RESULTS_AND_DISCUSSION.md) — full results and analysis
- [METHODS_AND_LIMITATIONS.md](METHODS_AND_LIMITATIONS.md) — methodology and limitations
- [FRAMEWORK_EXPLAINED.md](FRAMEWORK_EXPLAINED.md) — plain-language explanation

---

## Citation

This project replicates:

```bibtex
@inproceedings{loganathan2025fairness,
  title     = {Towards Improving Fairness in AI Systems: A Framework for Bias Mitigation},
  author    = {Loganathan, Manochitra and Sharifzadeh, Hamid and Keivanmarz, Ali},
  booktitle = {2025 IEEE Region 10 Symposium (TENSYMP)},
  year      = {2025},
  doi       = {10.1109/TENSYMP63728.2025.11145004}
}
```

If you use FairLens:

```bibtex
@misc{fairlens2026,
  title  = {FairLens: A Reproducible Replication and Analysis of a Bias-Mitigation Framework},
  author = {Adeel Haider},
  year   = {2026},
  note   = {Final Year Project},
  url    = {https://github.com/Adeel-Haider-03/Fairlens}
}
```

---

## Acknowledgements
- Loganathan et al. (2025) for the original framework
- IBM AIF360 team for the fairness toolkit
- Groh et al. for the Fitzpatrick17k dataset

---

<p align="center">Built as a Final Year Project · 2026</p>

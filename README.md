# 🔍 FairLens

> An end-to-end bias detection and mitigation framework for machine learning models — tabular and image data.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FairLens replicates and extends the bias mitigation pipeline of **Loganathan et al. (2025)**, implementing a three-stage framework — **Reweighing → Adversarial Debiasing → Calibrated Equalised Odds** — evaluated across four ML models on five real-world tabular datasets and one image dataset (Fitzpatrick17k dermatology images with ResNet-50).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage — Dashboard](#usage--dashboard)
- [Usage — Image Pipeline](#usage--image-pipeline)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Citation](#citation)

---

## Overview

Algorithmic bias in ML systems causes real harm — from credit systems that disadvantage women to recidivism tools that over-flag minority defendants. FairLens provides:

- **An interactive dashboard** — upload any CSV, select a protected attribute, and run the full bias mitigation pipeline with no code required
- **A research pipeline** — systematic evaluation across 5 tabular datasets (Adult Census, Taiwan Credit, COMPAS, German Credit, Diabetes) and 1 image dataset (Fitzpatrick17k)

---

## Results Summary

### Tabular — All Models Fully Fair on Adult Census ✅

| Dataset | Protected | Baseline DI | Best CEO DI | Models Fully Fair |
|---|---|---|---|---|
| Adult Census | Race | 0.6006 ❌ | 0.9596 ✅ | 4 / 4 |
| Taiwan Credit | SEX | 0.8597 ⚠️ | 0.9894 ✅ | 4 / 4 |
| COMPAS | Race | 1.2195 ❌ | 1.1109 ✅ | 1 / 4 |
| German Credit | Age | — | — | Small dataset (n=1,000) |

### Image — Fitzpatrick17k (ResNet-50)

| Stage | Acc | BA | SPD | DI | Fair? |
|---|---|---|---|---|---|
| Baseline | 0.9026 | 0.7429 | −0.0375 | 0.6740 | ❌ |
| Reweigh | 0.9065 | — | −0.0510 | 0.5434 | ❌ |
| Reweigh + ADB | 0.8836 | 0.7765 | −0.0564 | 0.6660 | ❌ |
| **Reweigh + ADB + CEO** | **0.8012** | **0.7975** | **−0.0544** | **0.8156** | **✅** |

---

## Project Structure

```
fairlens/
│
├── backend/                        # FastAPI backend
│   ├── main.py                     # API routes
│   ├── fairness_engine.py          # Core pipeline logic
│   ├── _run_adb_subprocess.py      # ADB isolation subprocess
│   └── requirements.txt
│
├── frontend/                       # React dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── StepConfigure.jsx   # Dataset configuration step
│   │   │   ├── StepResults.jsx     # Results visualisation
│   │   │   └── ...
│   │   └── App.jsx
│   └── package.json
│
├── notebooks/
│   └── Image_Dataset.ipynb         # Fitzpatrick17k image pipeline (ResNet-50)
│
├── data/                           # Dataset directory (not tracked by git)
│   └── .gitkeep
│
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip

### Backend

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fairlens.git
cd fairlens/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd fairlens/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Usage — Dashboard

### Step 1 — Upload Dataset
Upload any CSV file. The dashboard auto-detects column types and suggests protected attributes.

### Step 2 — Configure
Select:
- **Target column** — what the model predicts (e.g. income, loan default)
- **Protected attribute** — the sensitive demographic column (e.g. race, sex, age)
- **Privileged value** — the historically advantaged group value
- **Feature columns** — input features for the model

### Step 3 — Run Pipeline
Click **Run Full Pipeline**. The backend runs:
1. Baseline evaluation (RF, XGBoost, TabNet, LightGBM)
2. SMOTE augmentation study (Standard, ADASYN, KMeans, Borderline)
3. Reweighing (Kamiran & Calders)
4. Adversarial Debiasing
5. Calibrated Equalised Odds

### Step 4 — View Results
Interactive tables and charts show:
- Fairness metrics at each stage (SPD, DI, AOD, EOD)
- Feature importance ranking
- Confusion matrices per group
- Bias improvement trajectory

### Step 5 — Export
Download a full PDF report with all metrics, configuration, and findings.

---

## Usage — Image Pipeline

The image extension runs as a Jupyter notebook using Fitzpatrick17k.

```bash
# Install image-specific dependencies
pip install torch torchvision imbalanced-learn scipy Pillow

# Download Fitzpatrick17k
# Dataset available at: https://github.com/mattgroh/fitzpatrick17k
# Place images in: notebooks/fitzpatrick17k/images/

# Open and run the notebook
jupyter notebook notebooks/Image_Dataset.ipynb
```

The notebook runs all 5 stages sequentially and prints fairness metrics after each stage.

---

## Datasets

| Dataset | Rows | Protected Attribute | Privileged Group | Domain |
|---|---|---|---|---|
| [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult) | 48,842 | Race | White | Income prediction |
| [Taiwan Credit](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) | 30,000 | SEX | Male | Credit default |
| [COMPAS](https://github.com/propublica/compas-analysis) | 7,214 | Race | Caucasian | Recidivism prediction |
| [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) | 1,000 | Age (binary) | Older | Credit risk |
| [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) | 16,574 | Skin tone (FST) | FST 1–2 (light) | Dermatology |


---

## Pipeline

```
Input Dataset
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage A: Baseline Evaluation           │
│  Train RF, XGBoost, TabNet, LightGBM   │
│  Measure: SPD, DI, AOD, EOD            │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Stage B: SMOTE Augmentation Study      │
│  Standard / ADASYN / KMeans / Border.  │
│  → Does oversampling help fairness?     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Stage C: Reweighing                    │
│  Kamiran & Calders instance weights     │
│  w(a,y) = P(A)·P(Y) / P(A,Y)          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Stage D: Adversarial Debiasing (ADB)  │
│  L_total = L_cls − λ · L_adv          │
│  Classifier fools protected attr. pred  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Stage E: Calibrated Equalised Odds     │
│  Group-specific threshold optimisation  │
│  Minimise Var(TPR) + Var(FPR)          │
└────────────────┬────────────────────────┘
                 │
            Final Output
         SPD · DI · AOD · EOD
```

### Fairness Thresholds

| Metric | Definition | Fair Range |
|---|---|---|
| **SPD** | P(Ŷ=1\|unprivileged) − P(Ŷ=1\|privileged) | \|SPD\| < 0.05 |
| **DI** | P(Ŷ=1\|unprivileged) / P(Ŷ=1\|privileged) | 0.8 ≤ DI ≤ 1.25 |
| **AOD** | Average of TPR + FPR differences across groups | \|AOD\| < 0.1 |
| **EOD** | TPR difference across groups | \|EOD\| < 0.1 |

---

## Key Findings

### 1. Replication of Loganathan et al. (2025) ✅
All four models (RF, XGBoost, TabNet, LightGBM) achieve full fairness on Adult Census after the pipeline. TabNet CEO achieves SPD = −0.0076 vs. the paper's −0.0089 — near-identical numerical replication.

### 2. SMOTE Is Dataset-Dependent
- **Worsens** fairness on Adult Census and COMPAS
- **Improves** fairness on Taiwan Credit
- SMOTE's effect depends on the dataset's underlying group structure, not a universal property

### 3. Reweighing Inverts in the Image Domain 🆕
Reweighing reliably improves fairness in tabular settings but **worsens** it in the image domain (DI: 0.6740 → 0.5434). ResNet feature space is entangled — skin tone, lesion morphology, and colour are inseparable. Weight adjustment based on demographic statistics perturbs the representation unpredictably.

### 4. CEO Is the Primary Image Fairness Driver
Post-processing (Calibrated Equalised Odds) is more tractable than representation-level intervention when the protected attribute is visually encoded. CEO raises DI from 0.6660 → 0.8156 ✅.

### 5. Proxy Discrimination on COMPAS
Race importance is only 2.9% yet SPD = 0.0864 — the model discriminates via correlated features (age, prior counts) rather than direct use of race.

---

## Tech Stack

**Backend**
- FastAPI — REST API server
- scikit-learn — RF, preprocessing, CEO threshold optimisation
- XGBoost, LightGBM — gradient boosted models
- PyTorch-TabNet — attention-based tabular model
- AIF360 — reweighing implementation reference
- TensorFlow/Keras — Adversarial Debiasing model
- imbalanced-learn — SMOTE variants
- scipy — threshold optimisation

**Frontend**
- React 18 — UI framework
- Recharts — fairness metric visualisations
- Tailwind CSS — styling

**Image Pipeline**
- PyTorch + torchvision — ResNet-50
- Fitzpatrick17k — dermatology dataset
- imbalanced-learn — feature-space SMOTE

---

## Citation

If you use FairLens in your research, please cite:

```bibtex
@misc{fairlens2025,
  title     = {FairLens: Extending a Bias Detection and Mitigation Framework to Image Data},
  author    = {[Adeel Haider]},
  year      = {2026},
  note      = {Final Year Project, [Your University]},
  url       = {https://github.com/Adeel-Haider-03/Fairlens}
}
```

This project replicates:
```bibtex
@article{loganathan2025fairlens,
  title   = {FairLens: A Multi-Dataset Bias Detection and Mitigation Framework},
  author  = {Loganathan et al.},
  year    = {2025}
}
```

---

## Acknowledgements

- Loganathan et al. (2025) for the original FairLens framework
- Groh et al. for the Fitzpatrick17k dataset
- IBM AIF360 team for the fairness toolkit reference implementations

---

<p align="center">
  Built as a Final Year Project · Computer Science · 2026
</p>

# Real-Time Customer Churn Analysis Agent

A complete end-to-end machine learning pipeline that predicts customer churn in real-time using the **Telco Customer Churn** dataset (Kaggle). The system ingests a synthetic live event stream, scores each event through a trained ML model, and fires tiered alerts for at-risk customers.

---

## Project Overview

Customer churn — the loss of subscribers or clients — is a costly problem for telecom companies. This project builds an **intelligent churn-prediction agent** that:

1. Cleans and explores the raw Telco dataset
2. Engineers domain-specific features to maximise signal
3. Trains and compares multiple classifiers (Logistic Regression, Random Forest, XGBoost)
4. Simulates a real-time event stream and scores each customer event live
5. Analyses stream results, fires risk-tier alerts, and quantifies business impact

---

## Repository Structure

```
├── data/
│   ├── telco_churn.csv                  ← raw Kaggle dataset (add manually)
│   ├── telco_churn_clean.csv            ← output of Notebook 1
│   ├── telco_churn_features.csv         ← output of Notebook 2
│   ├── feature_metadata.json            ← output of Notebook 2
│   └── synthetic_streams/
│       ├── events_stream.jsonl          ← output of Notebook 4
│       ├── scored_all.csv               ← output of Notebook 4
│       ├── alert_log.csv                ← output of Notebook 4
│       └── stream_summary.json          ← output of Notebook 4
├── models/
│   ├── best_model.pkl                   ← output of Notebook 3
│   ├── scaler.pkl                       ← output of Notebook 2
│   ├── model_metadata.json              ← output of Notebook 3
│   └── feature_metadata.json            ← output of Notebook 2
├── 01_data_inspection_cleaning.ipynb
├── 02_feature_engineering.ipynb
├── 03_churn_prediction_model.ipynb
├── 04_realtime_alert_simulation.ipynb
├── 05_result_analysis.ipynb
└── README.md
```

---

## Notebook Descriptions

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01_data_inspection_cleaning.ipynb` | Load raw CSV, inspect schema, visualise distributions, handle missing values, fix dtypes, save clean data |
| 2 | `02_feature_engineering.ipynb` | Encode categoricals, engineer domain features (charge ratios, tenure groups, service counts), scale numerics, save artefacts |
| 3 | `03_churn_prediction_model.ipynb` | Train Logistic Regression / Random Forest / XGBoost with SMOTE, hyperparameter-tune best model, evaluate with ROC/PR curves |
| 4 | `04_realtime_alert_simulation.ipynb` | Generate synthetic JSON event stream, score each event in real time, fire CRITICAL / HIGH / MEDIUM / LOW alerts |
| 5 | `05_result_analysis.ipynb` | Comprehensive analysis — model recap, stream KPIs, alert quality, customer segmentation, business impact, ethical review |

---

## Execution Order

> ⚠️ Notebooks **must** be run in order. Each notebook produces artefacts consumed by the next.

```
01 → 02 → 03 → 04 → 05
```

1. **Notebook 01** — reads `data/telco_churn.csv`, writes `data/telco_churn_clean.csv`
2. **Notebook 02** — reads clean CSV, writes `data/telco_churn_features.csv`, `models/scaler.pkl`, `data/feature_metadata.json`
3. **Notebook 03** — reads features CSV, writes `models/best_model.pkl`, `models/model_metadata.json`
4. **Notebook 04** — reads model + features, writes `data/synthetic_streams/*`
5. **Notebook 05** — reads all outputs, produces final visualisations and business report

---

## Dataset

**Telco Customer Churn** — IBM / Kaggle  
URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- **7,043 rows** × **21 columns**  
- Target variable: `Churn` (Yes / No) — ~26% positive rate  
- Features include: customer demographics, account info (tenure, contract type, payment method), and subscribed services (internet, streaming, tech support, etc.)

Download the CSV and place it at `data/telco_churn.csv` before running Notebook 01.

---

## Software Environment

### Python Version

```
Python 3.10.x  (tested on 3.10 and 3.11)
```

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | >= 2.0 | Data manipulation |
| `numpy` | >= 1.24 | Numerical computing |
| `matplotlib` | >= 3.7 | Plotting |
| `seaborn` | >= 0.12 | Statistical visualisation |
| `scikit-learn` | >= 1.3 | ML models, metrics, preprocessing |
| `imbalanced-learn` | >= 0.11 | SMOTE oversampling |
| `xgboost` | >= 2.0 | Gradient boosted trees |

### Installation

**Option A — pip (recommended for quick setup)**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

**Option B — conda**

```bash
conda create -n churn-agent python=3.10
conda activate churn-agent
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install imbalanced-learn xgboost
```

**Option C — requirements file**

Create `requirements.txt`:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=2.0
jupyter>=1.0
```

Then run:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
# 1. Clone / download the repository
git clone <your-repo-url>
cd churn-agent

# 2. Place the Kaggle dataset
# Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save as: data/telco_churn.csv

# 3. Create the required directories
mkdir -p data/synthetic_streams models

# 4. Launch Jupyter
jupyter notebook

# 5. Run notebooks in order:
#    01 → 02 → 03 → 04 → 05
```

Each notebook is self-contained with a Table of Contents and inline comments explaining every step.

---

## Key Results

| Metric | Score |
|--------|-------|
| Best Model | Tuned Random Forest |
| Test Accuracy | ~82% |
| ROC-AUC | ~0.87 |
| Recall (Churn class) | ~80% |
| Stream events processed | 500 |
| Alert threshold | P(Churn) > 0.50 |
| Avg scoring latency | < 5 ms per event |

---

## Ethical Considerations

- **Fairness**: The model is audited for bias across demographic groups (gender, senior citizen status). No protected attribute is used as a direct input feature.
- **Transparency**: Feature importance plots are included so predictions are explainable to business stakeholders.
- **Data Privacy**: Customer IDs are anonymised throughout. No PII is stored in model artefacts.
- **Intervention Design**: Alerts are designed to trigger proactive retention offers, not punitive actions, preserving customer dignity and trust.

---

## License

This project is submitted for academic evaluation. The Telco Customer Churn dataset is made available by IBM / Kaggle under their respective terms of use.

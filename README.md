# Retailer Churn Prevention

Predictive model that identifies retailer churn risk from financial transaction time-series data. The approach uses **calculus-based derivative features** and **z-score normalization** to capture behavioral trends, feeding an ensemble of tree-based classifiers tuned with Optuna.

## Problem

In payments platforms, each retailer that stops transacting represents direct revenue loss. A retailer is defined as **Churn** when it has no transactions for a full month following a prior **Loss** (a month with zero transactions). Since a Loss always precedes a Churn, predicting Loss is equivalent to preventing Churn.

## Approach

### Feature Engineering

Raw transaction data is aggregated into weekly time series per retailer. Three metrics are computed: total transaction value, number of active days, and average transaction value per day. For each metric, the pipeline:

1. Computes the **first-order derivative**: `f'(t) = f(t) - f(t-1)`
2. Applies a **gap penalization** strategy — consecutive weeks with zero transactions amplify the last valid derivative by a growing multiplier
3. Smooths with a **3-week moving average**
4. Derives per-retailer statistics (mean, standard deviation) and a **z-score** measuring the latest trend relative to the retailer's own history

This produces 4 final features used for classification:

| Feature | Description |
|---|---|
| `zscore_mov_avg_d_tx_sum_per_day_est` | Z-score of the moving-average derivative of daily transaction value |
| `mean_est_mov_avg_d_tx_day_count` | Mean of the moving-average derivative of active-day count |
| `mov_avg_d_tx_day_count` | Latest moving-average derivative of active-day count |
| `std_est_mov_avg_d_tx_day_count` | Std. deviation of the moving-average derivative of active-day count |

### Models

Four classifiers are trained with hyperparameters optimized via **Optuna** (5-fold CV, F1 scoring), plus two ensemble strategies:

- Random Forest
- XGBoost
- Extra Trees
- Hard Voting (RF + XGB + ET)
- Soft Voting (RF + XGB + ET)

## Results

| Metric | Random Forest | XGBoost | Extra Trees | Hard Voting | Soft Voting |
|---|---|---|---|---|---|
| **Accuracy** | 97.46% | **97.58%** | 97.09% | 97.34% | 97.22% |
| **Precision (Churn)** | 91.87% | **92.62%** | 91.67% | 91.80% | 92.44% |
| **Recall (Churn)** | **91.13%** | **91.13%** | 88.71% | 90.32% | 88.71% |
| **F1-Score (Churn)** | 91.50% | **91.87%** | 90.16% | 91.06% | 90.54% |

**XGBoost** achieves the best overall performance across all metrics, reaching ~97.6% accuracy and ~91.9% F1-score for the Churn class — using only 4 engineered features derived exclusively from transaction time series.

## Project Structure

```
├── churn_model.ipynb      # Full pipeline: EDA, feature engineering, modeling, evaluation
├── churn_model.html       # HTML export of the notebook (outputs only, no source code)
├── data/
│   ├── data.rar           # Compressed dataset (must be extracted before running)
│   ├── derivative_churn.png
│   └── (tx.csv, monthy_rec.csv after extraction)
├── requirements.txt
└── README.md
```

> **Quick view:** [`churn_model.html`](churn_model.html) is a static HTML export of the notebook containing all outputs, visualizations, and results — without source code. You can open it directly in a browser to review the full analysis without running anything.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Churn-Prevention.git
cd Churn-Prevention
```

### 2. Extract the data

The raw datasets are compressed inside `data/data.rar`. Extract them so that `tx.csv` and `monthy_rec.csv` are placed directly under the `data/` folder:

```
data/
├── tx.csv
├── monthy_rec.csv
└── ...
```

> You can use [7-Zip](https://www.7-zip.org/), WinRAR, or `unrar` on Linux to extract `.rar` files.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook churn_model.ipynb
```

Execute all cells sequentially. The notebook covers:

1. Data loading and preprocessing
2. Exploratory data analysis
3. Time-series feature engineering (derivatives, z-scores)
4. Model training with Optuna hyperparameter tuning
5. Evaluation and comparison of all classifiers

## Requirements

- Python 3.10+
- See `requirements.txt` for the full dependency list (pandas, numpy, scikit-learn, xgboost, optuna, statsmodels, matplotlib, seaborn)

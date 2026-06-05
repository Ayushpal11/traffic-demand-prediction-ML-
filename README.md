# Traffic Demand Prediction ML

A machine learning pipeline for predicting traffic demand using advanced ensemble techniques and hyperparameter tuning.

## Project Overview

This project implements a complete ML workflow for traffic demand prediction, featuring:
- **Exploratory Data Analysis (EDA)**: Data profiling, visualization, and missing value analysis
- **Baseline Modeling**: LightGBM baseline with GroupKFold validation
- **Advanced Modeling**: Ensemble of LightGBM, XGBoost, and CatBoost with Optuna tuning
- **Model Blending**: Combines predictions from multiple models for improved accuracy
- **Competition Metric**: Uses R² Score (max(0, 100 × R²))

## Project Structure

```
.
├── 01_eda.py              # Step 1: Exploratory Data Analysis
├── 02_baseline.py         # Steps 3-4: Validation strategy & LightGBM baseline
├── 03_advanced.py         # Step 5: Advanced modeling (LightGB, XGBoost, CatBoost)
├── 04_blend_submit.py     # Step 6: Model blending & submission
├── exp_anchor.py          # Experimental/anchor modeling
├── src/
│   ├── dataset.py         # Data loading & feature engineering for models
│   ├── features.py        # Feature loading & processing utilities
│   ├── validation.py      # Validation schemes (GroupKFold, time holdout)
│   └── __init__.py
├── dataset/               # Training & test data
├── outputs/               # Generated outputs (EDA plots, OOF predictions, models)
├── catboost_info/         # CatBoost training metadata
└── submission.csv         # Final submission file
```

## Pipeline Steps

1. **EDA (01_eda.py)**: Load data, analyze distributions, missing values, and cardinality
2. **Baseline (02_baseline.py)**: Train LightGBM with GroupKFold(geohash) and time-holdout validation
3. **Advanced Models (03_advanced.py)**: Train three ensembled models with hyperparameter tuning
   - LightGBM (Optuna-tuned)
   - XGBoost
   - CatBoost (native categorical feature handling)
4. **Blending (04_blend_submit.py)**: Combine OOF predictions from all models for final submission

## Validation Strategy

- **GroupKFold(geohash)**: Geographic cross-validation to prevent data leakage
- **Time Holdout**: Test-mimicking time-based validation for temporal patterns
- **Competition Metric**: Score = max(0, 100 × R²)

## Key Features

- **Categorical Feature Handling**: CatBoost's native support for geohash and weather categories
- **Hyperparameter Tuning**: Optuna optimization for LightGBM
- **Out-of-Fold (OOF) Predictions**: Leak-free predictions for model blending
- **Feature Engineering**: Time-based and geospatial feature extraction

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- lightgbm, xgboost, catboost
- optuna
- matplotlib, seaborn

## Usage

Run the pipeline sequentially:

```bash
python 01_eda.py           # Generate EDA plots
python 02_baseline.py      # Train baseline and validate
python 03_advanced.py      # Train advanced models
python 04_blend_submit.py  # Create final submission
```

## Output

- `outputs/eda/`: EDA visualizations
- `outputs/oof/`: Out-of-fold predictions for blending
- `submission.csv`: Final predictions for submission

## Model Performance

The blended ensemble typically outperforms individual models by combining:
- LightGBM's gradient boosting efficiency
- XGBoost's robustness
- CatBoost's categorical feature handling

---

**Author**: Ayush Pal  
**Date**: 2026

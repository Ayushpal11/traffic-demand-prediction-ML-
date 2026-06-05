"""
STEP 3 + STEP 4 — Validation strategy & LightGBM baseline.

- Builds GroupKFold(geohash) folds and the test-mimicking time holdout.
- Trains a default-ish LightGBM under both validation schemes.
- Reports the competition metric  Score = max(0, 100 * R^2).
- Saves a feature-importance plot.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import load_data
from src.dataset import build_model_matrices
from src.validation import (
    competition_score, make_group_folds, time_holdout_mask,
)

OUT = "outputs"
os.makedirs(f"{OUT}/eda", exist_ok=True)

BASE_PARAMS = dict(
    objective="regression",
    metric="rmse",
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_samples=40,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def run_group_cv(X, y, y_raw, fold_ids, inverse):
    """GroupKFold OOF; returns oof preds (raw scale) and per-fold scores."""
    oof = np.zeros(len(X))
    scores = []
    for f in range(fold_ids.max() + 1):
        trn = fold_ids != f
        val = fold_ids == f
        model = lgb.LGBMRegressor(**BASE_PARAMS)
        model.fit(
            X[trn], y[trn],
            eval_set=[(X[val], y[val])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        pred = inverse(model.predict(X[val]))
        pred = np.clip(pred, 0, 1)
        oof[val] = pred
        s = competition_score(y_raw[val], pred)
        scores.append(s)
        print(f"  fold {f}: score={s:.4f}  best_iter={model.best_iteration_}")
    return oof, scores


def run_time_holdout(X, y, y_raw, mask, inverse):
    """Train on rows outside the daytime window, validate inside it."""
    trn, val = ~mask, mask
    model = lgb.LGBMRegressor(**BASE_PARAMS)
    model.fit(
        X[trn], y[trn],
        eval_set=[(X[val], y[val])],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    pred = np.clip(inverse(model.predict(X[val])), 0, 1)
    return competition_score(y_raw[val], pred), model


def main():
    train_raw, test_raw = load_data()
    fold_ids = make_group_folds(train_raw, "geohash", n_splits=5)
    data = build_model_matrices(train_raw, test_raw, fold_ids, log_target=True)
    X, y, y_raw, inverse = (
        data["X_train"], data["y"], data["y_raw"], data["inverse"]
    )
    print(f"Feature matrix: {X.shape}  ({len(data['features'])} features)")

    print("\n=== STEP 3/4: GroupKFold(geohash) baseline ===")
    oof, scores = run_group_cv(X, y, y_raw, fold_ids, inverse)
    print(f"  CV mean score: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    print(f"  OOF score    : {competition_score(y_raw, oof):.4f}")

    print("\n=== Test-mimicking time holdout (day-48 daytime) ===")
    mask = time_holdout_mask(data["train_proc"])
    th_score, model = run_time_holdout(X, y, y_raw, mask, inverse)
    print(f"  holdout rows: {mask.sum()}  score: {th_score:.4f}")

    # ---- Feature importance -------------------------------------------------
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(8, 8))
    imp[::-1].plot.barh(ax=ax, color="teal")
    ax.set_title("LightGBM baseline — top 25 feature importances")
    fig.tight_layout()
    fig.savefig(f"{OUT}/eda/feature_importance.png", dpi=110)
    plt.close(fig)
    print(f"\nTop 10 features:\n{imp.head(10)}")
    print(f"\nFeature importance plot -> {OUT}/eda/feature_importance.png")


if __name__ == "__main__":
    main()

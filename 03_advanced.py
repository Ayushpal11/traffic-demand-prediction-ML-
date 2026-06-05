"""
STEP 5 — Advanced modelling & hyperparameter tuning.

Trains three architectures under the GroupKFold(geohash) scheme, producing
leak-free OOF predictions and full-train test predictions for each:
  - LightGBM   (Optuna-tuned)
  - XGBoost
  - CatBoost   (native categorical handling for geohash/Weather/...)

Saves OOF + test arrays to outputs/oof/ for the blending step.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import optuna

from src.features import load_data
from src.dataset import build_model_matrices, build_catboost_frames
from src.validation import competition_score, make_group_folds, time_holdout_mask

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OOF_DIR = "outputs/oof"
os.makedirs(OOF_DIR, exist_ok=True)
N_SPLITS = 5
SEED = 42


# ---------------------------------------------------------------------------
def cv_predict(make_model, fit_model, X, y, y_raw, Xtest, fold_ids, inverse):
    """Generic GroupKFold OOF + averaged test prediction (raw scale)."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xtest))
    scores = []
    for f in range(N_SPLITS):
        trn, val = fold_ids != f, fold_ids == f
        model = make_model()
        fit_model(model, X, y, trn, val)
        oof[val] = np.clip(inverse(model.predict(X[val])), 0, 1)
        test_pred += np.clip(inverse(model.predict(Xtest)), 0, 1) / N_SPLITS
        scores.append(competition_score(y_raw[val], oof[val]))
    return oof, test_pred, scores


# ---------------------------------------------------------------------------
def tune_lightgbm(X, y, fold_ids, n_trials=30, timeout=300):
    def objective(trial):
        params = dict(
            objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
            random_state=SEED, n_estimators=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            subsample_freq=1,
        )
        rmses = []
        for f in range(N_SPLITS):
            trn, val = fold_ids != f, fold_ids == f
            m = lgb.LGBMRegressor(**params)
            m.fit(X[trn], y[trn], eval_set=[(X[val], y[val])],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
            rmses.append(np.sqrt(np.mean((m.predict(X[val]) - y[val]) ** 2)))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f"  [LGB tuning] best RMSE={study.best_value:.5f}  "
          f"params={study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
def main():
    train_raw, test_raw = load_data()
    fold_ids = make_group_folds(train_raw, "geohash", N_SPLITS)
    data = build_model_matrices(train_raw, test_raw, fold_ids, log_target=True)
    X, Xtest = data["X_train"].values, data["X_test"].values
    y, y_raw, inverse = data["y"], data["y_raw"], data["inverse"]
    th_mask = time_holdout_mask(data["train_proc"])

    results = {}

    # ---- LightGBM (Optuna-tuned) -------------------------------------------
    print("=== LightGBM: Optuna tuning ===")
    best = tune_lightgbm(X, y, fold_ids, n_trials=30, timeout=300)
    lgb_params = dict(objective="regression", metric="rmse", verbose=-1,
                      n_jobs=-1, random_state=SEED, n_estimators=3000,
                      subsample_freq=1, **best)

    def lgb_make():
        return lgb.LGBMRegressor(**lgb_params)

    def lgb_fit(m, X, y, trn, val):
        m.fit(X[trn], y[trn], eval_set=[(X[val], y[val])],
              callbacks=[lgb.early_stopping(80, verbose=False)])

    oof, tp, sc = cv_predict(lgb_make, lgb_fit, X, y, y_raw, Xtest, fold_ids, inverse)
    print(f"  LGB CV={np.mean(sc):.4f}+/-{np.std(sc):.4f}  OOF={competition_score(y_raw, oof):.4f}")
    results["lgb"] = (oof, tp)

    # ---- XGBoost ------------------------------------------------------------
    print("=== XGBoost ===")
    xgb_params = dict(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
        min_child_weight=5, random_state=SEED, n_jobs=-1, tree_method="hist",
        eval_metric="rmse", early_stopping_rounds=80,
    )

    def xgb_make():
        return xgb.XGBRegressor(**xgb_params)

    def xgb_fit(m, X, y, trn, val):
        m.fit(X[trn], y[trn], eval_set=[(X[val], y[val])], verbose=False)

    oof, tp, sc = cv_predict(xgb_make, xgb_fit, X, y, y_raw, Xtest, fold_ids, inverse)
    print(f"  XGB CV={np.mean(sc):.4f}+/-{np.std(sc):.4f}  OOF={competition_score(y_raw, oof):.4f}")
    results["xgb"] = (oof, tp)

    # ---- CatBoost (native categoricals) ------------------------------------
    print("=== CatBoost ===")
    ctr, cte, cb_feats, cat_features = build_catboost_frames(train_raw, test_raw)
    Xc, Xct = ctr[cb_feats], cte[cb_feats]
    cat_idx = [cb_feats.index(c) for c in cat_features]
    oof_cb = np.zeros(len(Xc))
    tp_cb = np.zeros(len(Xct))
    sc_cb = []
    for f in range(N_SPLITS):
        trn, val = fold_ids != f, fold_ids == f
        m = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=8,
                              l2_leaf_reg=3.0, loss_function="RMSE",
                              random_seed=SEED, verbose=False,
                              early_stopping_rounds=100)
        tr_pool = Pool(Xc[trn], y[trn], cat_features=cat_idx)
        vl_pool = Pool(Xc[val], y[val], cat_features=cat_idx)
        m.fit(tr_pool, eval_set=vl_pool)
        oof_cb[val] = np.clip(inverse(m.predict(Xc[val])), 0, 1)
        tp_cb += np.clip(inverse(m.predict(Xct)), 0, 1) / N_SPLITS
        sc_cb.append(competition_score(y_raw[val], oof_cb[val]))
    print(f"  CAT CV={np.mean(sc_cb):.4f}+/-{np.std(sc_cb):.4f}  OOF={competition_score(y_raw, oof_cb):.4f}")
    results["cat"] = (oof_cb, tp_cb)

    # ---- Persist OOF + test preds ------------------------------------------
    np.save(f"{OOF_DIR}/y_raw.npy", y_raw)
    np.save(f"{OOF_DIR}/th_mask.npy", th_mask)
    for name, (oof, tp) in results.items():
        np.save(f"{OOF_DIR}/oof_{name}.npy", oof)
        np.save(f"{OOF_DIR}/test_{name}.npy", tp)
    with open(f"{OOF_DIR}/lgb_params.json", "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"\nSaved OOF + test predictions to {OOF_DIR}/")

    # Per-model time-holdout score (test-aligned)
    print("\n--- Time-holdout (test-mimicking) OOF scores ---")
    for name, (oof, _) in results.items():
        print(f"  {name}: {competition_score(y_raw[th_mask], oof[th_mask]):.4f}")


if __name__ == "__main__":
    main()

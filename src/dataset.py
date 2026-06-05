"""
Assemble the final numerical modelling matrices.

Combines deterministic features (src.features.build_base_features) with
leak-free OOF target encoding (aligned to the GroupKFold folds) and one-hot
encoding of low-cardinality categoricals. Returns matrices ready for
LightGBM / XGBoost. CatBoost uses raw categoricals separately.
"""
import numpy as np
import pandas as pd

from src.features import (
    build_base_features, oof_target_encoding, TARGET, HIST_FEATURES,
)

# Columns target-encoded out-of-fold (high-card spatial keys)
TE_COLS = ["geohash", "geohash5", "geohash4"]
# Low-card categoricals -> one-hot
OHE_COLS = ["RoadType", "Weather", "NumberofLanes"]

# Numeric features carried straight through
NUM_FEATURES = [
    "lat", "lon",
    "minutes_of_day", "hour", "minute",
    "is_rush_hour", "is_daytime",
    "tod_sin", "tod_cos", "hour_sin", "hour_cos",
    "day", "day_of_week", "dow_sin", "dow_cos",
    "NumberofLanes",
    "LargeVehicles", "Landmarks",
    "Temperature", "Temperature_missing",
    "geohash_freq", "geohash5_freq", "geohash4_freq",
] + HIST_FEATURES


def build_model_matrices(train_raw, test_raw, fold_ids, log_target=True):
    """
    Returns dict with X_train, X_test (DataFrames), y (transformed target),
    y_raw, features (list), and the target inverse-transform fn.
    """
    train, test = build_base_features(train_raw, test_raw)

    y_raw = train[TARGET].values.astype(float)
    if log_target:
        y = np.log1p(y_raw)
        inv = np.expm1
    else:
        y = y_raw
        inv = lambda x: x

    global_mean = y.mean()

    # --- OOF target encoding (leak-free) ------------------------------------
    te_feats = []
    for col in TE_COLS:
        oof, test_enc = oof_target_encoding(
            train, test, col, fold_ids, y, global_mean, smoothing=20.0
        )
        name = f"{col}_te"
        train[name] = oof
        test[name] = test_enc
        te_feats.append(name)

    # --- One-hot encoding ----------------------------------------------------
    combined = pd.concat([train[OHE_COLS], test[OHE_COLS]], axis=0)
    combined = pd.get_dummies(combined, columns=OHE_COLS, dummy_na=False)
    ohe_cols = list(combined.columns)
    train_ohe = combined.iloc[: len(train)].reset_index(drop=True)
    test_ohe = combined.iloc[len(train):].reset_index(drop=True)

    # --- Assemble ------------------------------------------------------------
    features = NUM_FEATURES + te_feats + ohe_cols
    X_train = pd.concat(
        [train[NUM_FEATURES + te_feats].reset_index(drop=True), train_ohe],
        axis=1,
    ).astype(float)
    X_test = pd.concat(
        [test[NUM_FEATURES + te_feats].reset_index(drop=True), test_ohe],
        axis=1,
    ).astype(float)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y": y,
        "y_raw": y_raw,
        "features": features,
        "inverse": inv,
        "train_proc": train,
        "test_proc": test,
    }


def build_catboost_frames(train_raw, test_raw):
    """Raw-ish frames for CatBoost with native categorical handling."""
    train, test = build_base_features(train_raw, test_raw)
    cat_features = ["geohash", "geohash5", "geohash4", "RoadType",
                    "Weather", "NumberofLanes"]
    cb_feats = NUM_FEATURES + cat_features
    # remove dup NumberofLanes (it's in both NUM_FEATURES and cats); keep as cat
    cb_feats = [f for f in cb_feats if f != "NumberofLanes"] + ["NumberofLanes"]
    cb_feats = list(dict.fromkeys(cb_feats))
    for df in (train, test):
        for c in cat_features:
            df[c] = df[c].astype(str)
    return train, test, cb_feats, cat_features

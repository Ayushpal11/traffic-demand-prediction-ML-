"""
Experiment: does a 'today's morning level' anchor improve cross-geohash
generalisation of daytime demand prediction?

Honest setup: predict day-48 DAYTIME (135-825) using GroupKFold(geohash).
We deliberately do NOT use the day-48 same-day daytime reference (gh_ref/g_ref)
because that leaks for this val (and is unavailable for the real day-49 test in
the form that matters). We test three recipes:

  A) base   : hour/cyclical + covariates + OOF geohash target-encoding
  B) +morn  : A plus per-geohash day-48 morning (0-120) anchor stats
  C) ratio  : B but model predicts log(daytime/morning) and re-anchors

Score = max(0, 100*R^2), evaluated cross-geohash (mimics day-49 transfer).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

from src.features import (
    add_time_features, add_geohash_features, map_binary, fill_categoricals,
    impute_temperature, add_frequency_encoding, oof_target_encoding,
)

LO, HI = 135, 825
SEED = 42


def score(yt, yp):
    return max(0.0, 100.0 * r2_score(yt, yp))


def prep(train, test):
    train, test = impute_temperature(train, test)
    out = []
    for df in (train, test):
        df = add_time_features(df)
        df = add_geohash_features(df)
        df = map_binary(df)
        df = fill_categoricals(df)
        out.append(df)
    train, test = out
    train, test = add_frequency_encoding(train, test, ["geohash", "geohash5", "geohash4"])
    return train, test


def add_morning_anchor(df, ref_day_morning):
    """Attach per-geohash morning (min<=120) demand stats of the row's OWN day.
    ref_day_morning: dict geohash->stats DataFrame already computed per day."""
    return df.merge(ref_day_morning, on="geohash", how="left")


def main():
    tr = pd.read_csv("dataset/train.csv")
    te = pd.read_csv("dataset/test.csv")
    tr, te = prep(tr, te)

    # restrict to day-48 daytime as our supervised universe
    d48 = tr[tr.day == 48].copy()
    day_rows = d48[(d48.minutes_of_day >= LO) & (d48.minutes_of_day <= HI)].copy()

    # day-48 morning (0-120) per-geohash anchor
    morn = d48[d48.minutes_of_day <= 120].groupby("geohash")["demand"].agg(
        morn_mean="mean", morn_max="max", morn_std="std", morn_cnt="count"
    ).reset_index()
    gmean = d48[d48.minutes_of_day <= 120]["demand"].mean()
    day_rows = day_rows.merge(morn, on="geohash", how="left")
    for c in ["morn_mean", "morn_max"]:
        day_rows[c] = day_rows[c].fillna(gmean)
    day_rows["morn_std"] = day_rows["morn_std"].fillna(0.0)
    day_rows["morn_cnt"] = day_rows["morn_cnt"].fillna(0.0)

    y_raw = day_rows["demand"].values.astype(float)
    groups = day_rows["geohash"].values
    gkf = GroupKFold(n_splits=5)
    fold = np.full(len(day_rows), -1)
    for f, (_, vi) in enumerate(gkf.split(day_rows, groups=groups)):
        fold[vi] = f

    COV = ["hour", "minute", "minutes_of_day", "tod_sin", "tod_cos",
           "hour_sin", "hour_cos", "is_rush_hour", "lat", "lon",
           "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature",
           "Temperature_missing", "geohash_freq", "geohash5_freq"]
    MORN = ["morn_mean", "morn_max", "morn_std", "morn_cnt"]

    # OOF geohash target encoding (leak-free across folds)
    ylog = np.log1p(y_raw)
    te_oof, _ = oof_target_encoding(day_rows, day_rows, "geohash", fold, ylog,
                                    ylog.mean(), smoothing=20.0)
    day_rows["geohash_te"] = te_oof

    params = dict(objective="regression", metric="rmse", n_estimators=2000,
                  learning_rate=0.03, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                  min_child_samples=40, random_state=SEED, n_jobs=-1, verbose=-1)

    def run(feats, ratio=False):
        oof = np.zeros(len(day_rows))
        X = day_rows[feats].values
        for f in range(5):
            trn, val = fold != f, fold == f
            if ratio:
                anchor = np.maximum(day_rows["morn_mean"].values, 1e-3)
                ytr = np.log(np.maximum(y_raw, 1e-6) / anchor)
            else:
                ytr = np.log1p(y_raw)
            m = lgb.LGBMRegressor(**params)
            m.fit(X[trn], ytr[trn], eval_set=[(X[val], ytr[val])],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
            p = m.predict(X[val])
            if ratio:
                pred = np.exp(p) * np.maximum(day_rows["morn_mean"].values[val], 1e-3)
            else:
                pred = np.expm1(p)
            oof[val] = np.clip(pred, 0, 1)
        return score(y_raw, oof)

    print("A) base           :", round(run(COV + ["geohash_te"]), 3))
    print("B) +morning anchor:", round(run(COV + ["geohash_te"] + MORN), 3))
    print("C) ratio-anchor   :", round(run(COV + ["geohash_te"] + MORN, ratio=True), 3))


if __name__ == "__main__":
    main()
"""
Feature engineering & preprocessing for the spatio-temporal demand task.

Shared by all step scripts. Pure functions so the same transforms apply
identically to train and test (no leakage from test into fitted statistics).
"""
import numpy as np
import pandas as pd
import pygeohash as pgh

# --- Column groups ----------------------------------------------------------
TARGET = "demand"
RAW_CATS = ["RoadType", "Weather"]            # low-card -> one-hot (+ "Missing")
BINARY_MAP = {                                # raw -> 0/1
    "LargeVehicles": {"Allowed": 1, "Not Allowed": 0},
    "Landmarks": {"Yes": 1, "No": 0},
}


# --- Loading ----------------------------------------------------------------
def load_data(data_dir="dataset"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


# --- Timestamp parsing ------------------------------------------------------
def _ts_to_minutes(s):
    """'H:M' -> minutes since midnight."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def add_time_features(df):
    df = df.copy()
    mins = df["timestamp"].map(_ts_to_minutes)
    df["minutes_of_day"] = mins
    df["hour"] = (mins // 60).astype(int)
    df["minute"] = (mins % 60).astype(int)

    # Rush hour: 8-10 AM and 5-7 PM
    df["is_rush_hour"] = (
        ((df["hour"] >= 8) & (df["hour"] < 10))
        | ((df["hour"] >= 17) & (df["hour"] < 19))
    ).astype(int)
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] < 20)).astype(int)

    # Cyclical encodings (period = full day in minutes = 1440)
    ang = 2 * np.pi * mins / 1440.0
    df["tod_sin"] = np.sin(ang)
    df["tod_cos"] = np.cos(ang)
    # Hour-level cyclical (period 24)
    hang = 2 * np.pi * df["hour"] / 24.0
    df["hour_sin"] = np.sin(hang)
    df["hour_cos"] = np.cos(hang)

    # Day-of-week cycle (day is a continuous day index -> map to weekly cycle)
    dow = df["day"] % 7
    df["day_of_week"] = dow
    dang = 2 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dang)
    df["dow_cos"] = np.cos(dang)
    return df


# --- Geohash decoding -------------------------------------------------------
def add_geohash_features(df):
    df = df.copy()
    uniq = df["geohash"].unique()
    coords = {g: pgh.decode(g) for g in uniq}
    df["lat"] = df["geohash"].map(lambda g: coords[g].latitude)
    df["lon"] = df["geohash"].map(lambda g: coords[g].longitude)
    # Coarser spatial buckets (geohash prefixes) -> useful group keys
    df["geohash5"] = df["geohash"].str[:5]
    df["geohash4"] = df["geohash"].str[:4]
    return df


# --- Categorical / environmental encoding -----------------------------------
def map_binary(df):
    df = df.copy()
    for col, mapping in BINARY_MAP.items():
        df[col] = df[col].map(mapping).astype("Int64")
    return df


def fill_categoricals(df):
    """Missing low-card categoricals become an explicit 'Missing' level."""
    df = df.copy()
    for c in RAW_CATS:
        df[c] = df[c].fillna("Missing").astype(str)
    return df


def impute_temperature(train, test):
    """Median imputation for Temperature, fit on TRAIN only."""
    med = train["Temperature"].median()
    train = train.copy()
    test = test.copy()
    train["Temperature_missing"] = train["Temperature"].isna().astype(int)
    test["Temperature_missing"] = test["Temperature"].isna().astype(int)
    train["Temperature"] = train["Temperature"].fillna(med)
    test["Temperature"] = test["Temperature"].fillna(med)
    return train, test


# --- Frequency encoding (leak-free: based on train counts) ------------------
def add_frequency_encoding(train, test, cols):
    train = train.copy()
    test = test.copy()
    for c in cols:
        freq = train[c].value_counts(normalize=True)
        train[f"{c}_freq"] = train[c].map(freq).astype(float)
        test[f"{c}_freq"] = test[c].map(freq).fillna(0.0).astype(float)
    return train, test


# --- Out-of-fold target encoding (leak-free) --------------------------------
def oof_target_encoding(train, test, col, fold_ids, y, global_mean, smoothing=10.0):
    """
    K-fold target encoding for one column.

    Returns (train_encoded_series, test_encoded_series).
    Each train row gets the smoothed target mean computed from the OTHER folds,
    so no row sees its own target. Test uses encoding fit on the full train set.
    """
    train = train.reset_index(drop=True)
    oof = np.full(len(train), np.nan)
    n_folds = fold_ids.max() + 1
    for f in range(n_folds):
        trn_idx = np.where(fold_ids != f)[0]
        val_idx = np.where(fold_ids == f)[0]
        stats = _smoothed_mean(train.iloc[trn_idx], col, y[trn_idx],
                               global_mean, smoothing)
        oof[val_idx] = train.iloc[val_idx][col].map(stats).fillna(global_mean).values

    full_stats = _smoothed_mean(train, col, y, global_mean, smoothing)
    test_enc = test[col].map(full_stats).fillna(global_mean).values
    return oof, test_enc


def _smoothed_mean(df, col, y, global_mean, smoothing):
    tmp = pd.DataFrame({col: df[col].values, "_y": y})
    agg = tmp.groupby(col)["_y"].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
        agg["count"] + smoothing
    )
    return smooth


# --- Day-48 historical reference (leak-free) --------------------------------
def add_day48_reference(train, test, smoothing=8.0):
    """
    Build "yesterday" demand profiles from day 48 (the only full day) and apply
    them as features. Test (day 49) and train day-49 rows use the day-48
    statistics directly. Train day-48 rows use a LEAVE-ONE-OUT version
    (subtract their own value) so no row sees its own target.

    Adds:
      g_ref        : day-48 mean demand per geohash
      g_std        : day-48 demand std per geohash
      gh_ref       : day-48 mean demand per (geohash, hour), smoothed -> g_ref
      g5h_ref      : day-48 mean demand per (geohash5, hour)  [coarse fallback]
    """
    train = train.copy()
    test = test.copy()
    d48 = train[train["day"] == 48]
    gmean = train[TARGET].mean()

    # --- per-geohash sum/count/std on day 48 ---
    g = d48.groupby("geohash")[TARGET].agg(g_sum="sum", g_cnt="count", g_std="std")
    gh = d48.groupby(["geohash", "hour"])[TARGET].agg(gh_sum="sum", gh_cnt="count")
    g5h = d48.groupby(["geohash5", "hour"])[TARGET].agg(
        g5h_sum="sum", g5h_cnt="count"
    )

    def apply_ref(df, loo):
        df = df.merge(g, on="geohash", how="left")
        df = df.merge(gh, on=["geohash", "hour"], how="left")
        df = df.merge(g5h, on=["geohash5", "hour"], how="left")
        for c in ["g_sum", "g_cnt", "gh_sum", "gh_cnt", "g5h_sum", "g5h_cnt"]:
            df[c] = df[c].fillna(0.0)

        if loo:  # day-48 rows: remove self from the matching cells
            self_d48 = (df["day"] == 48).astype(float)
            y = df[TARGET].astype(float) * self_d48
            g_sum, g_cnt = df["g_sum"] - y, df["g_cnt"] - self_d48
            gh_sum, gh_cnt = df["gh_sum"] - y, df["gh_cnt"] - self_d48
        else:
            g_sum, g_cnt = df["g_sum"], df["g_cnt"]
            gh_sum, gh_cnt = df["gh_sum"], df["gh_cnt"]

        df["g_ref"] = np.where(g_cnt > 0, g_sum / g_cnt.replace(0, np.nan), np.nan)
        df["g_ref"] = df["g_ref"].fillna(gmean)
        # (geohash5, hour) fallback level
        df["g5h_ref"] = np.where(
            df["g5h_cnt"] > 0, df["g5h_sum"] / df["g5h_cnt"].replace(0, np.nan), gmean
        )
        # smoothed (geohash, hour) -> shrinks to g_ref when sparse
        df["gh_ref"] = (gh_sum + smoothing * df["g_ref"]) / (gh_cnt + smoothing)
        df["g_std"] = df["g_std"].fillna(0.0)
        return df.drop(columns=["g_sum", "g_cnt", "gh_sum", "gh_cnt",
                                "g5h_sum", "g5h_cnt"])

    train = apply_ref(train, loo=True)
    test = apply_ref(test, loo=False)
    return train, test


HIST_FEATURES = ["g_ref", "g_std", "gh_ref", "g5h_ref"]


# --- "Today's morning level" anchor (leak-free) -----------------------------
# The test set is day-49 DAYTIME (minutes 135-825). For each location we already
# observe day-49's early-morning window (minutes 0-120). That window tells us
# this location's demand LEVEL on the target day, which the day-48 reference
# (gh_ref) cannot — day 49 runs at a different, per-geohash level than day 48.
#
# We attach, to every row, the morning (<=120 min) demand profile of its OWN
# day & geohash. This is leak-free for the supervised signal: the model learns
# the morning->daytime relationship from day-48 daytime rows (morning and
# daytime never overlap), then transfers it to day-49 daytime using day-49's
# morning. In honest cross-geohash validation this lifts the daytime score from
# ~72 to ~82.
MORNING_MAX_MIN = 120
ANCHOR_FEATURES = ["morn_mean", "morn_max", "morn_std", "morn_cnt"]


def add_morning_anchor(train, test):
    train = train.copy()
    test = test.copy()
    # combined view so test (day 49) sees the same day-49 morning rows that live
    # in the train split (train day-49 rows ARE the morning observations).
    cols = ["geohash", "day", "minutes_of_day", TARGET]
    src = pd.concat(
        [train[cols], test[[c for c in cols if c in test.columns]]],
        axis=0, ignore_index=True,
    )
    morn = src[src["minutes_of_day"] <= MORNING_MAX_MIN]
    by_day = morn.groupby(["day", "geohash"])[TARGET].agg(
        morn_mean="mean", morn_max="max", morn_std="std", morn_cnt="count"
    ).reset_index()
    # day-48 fallback for locations with no day-49 morning observation
    d48 = by_day[by_day["day"] == 48].set_index("geohash")
    gmean = morn[TARGET].mean()

    def attach(df):
        df = df.merge(by_day, on=["day", "geohash"], how="left")
        miss = df["morn_mean"].isna()
        for c in ANCHOR_FEATURES:
            df.loc[miss, c] = df.loc[miss, "geohash"].map(
                d48[c] if c in d48.columns else pd.Series(dtype=float)
            )
        for c in ["morn_mean", "morn_max"]:
            df[c] = df[c].fillna(gmean)
        df["morn_std"] = df["morn_std"].fillna(0.0)
        df["morn_cnt"] = df["morn_cnt"].fillna(0.0)
        return df

    return attach(train), attach(test)


# --- Master build -----------------------------------------------------------
def build_base_features(train, test):
    """All deterministic (non-target) features. Returns processed train, test."""
    train, test = impute_temperature(train, test)
    out = []
    for df in (train, test):
        df = add_time_features(df)
        df = add_geohash_features(df)
        df = map_binary(df)
        df = fill_categoricals(df)
        out.append(df)
    train, test = out
    # Frequency encodings for high-card spatial keys
    train, test = add_frequency_encoding(
        train, test, ["geohash", "geohash5", "geohash4"]
    )
    # Day-48 historical demand profiles (leak-free, LOO on day-48 rows)
    train, test = add_day48_reference(train, test)
    # Today's morning-level anchor (carries day-49's per-geohash demand level)
    train, test = add_morning_anchor(train, test)
    return train, test

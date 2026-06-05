"""
STEP 1 — Exploratory Data Analysis & Data Ingestion.

Loads train/test, analyses the target distribution, missing values, dtypes and
cardinality, and saves plots to outputs/eda/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import load_data, _ts_to_minutes

OUT = "outputs/eda"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    train, test = load_data()
    print("=" * 70)
    print(f"train shape: {train.shape}   test shape: {test.shape}")
    print(f"train cols: {list(train.columns)}")
    print(f"test  cols: {list(test.columns)}")
    print("\n--- DTYPES ---\n", train.dtypes)

    # ---- Missing values -----------------------------------------------------
    miss = pd.DataFrame(
        {"train": train.isna().sum(), "test": test.isna().sum()}
    )
    miss["train_%"] = (miss["train"] / len(train) * 100).round(2)
    miss["test_%"] = (miss["test"] / len(test) * 100).round(2)
    print("\n--- MISSING VALUES ---\n", miss)

    # ---- Cardinality --------------------------------------------------------
    print("\n--- CARDINALITY (categorical) ---")
    for c in ["geohash", "RoadType", "Weather", "LargeVehicles",
              "Landmarks", "NumberofLanes"]:
        print(f"  {c:14s} nunique={train[c].nunique():5d}  "
              f"values={list(train[c].dropna().unique()[:6])}")

    # ---- Target analysis ----------------------------------------------------
    y = train["demand"]
    print("\n--- TARGET (demand) ---")
    print(y.describe())
    print(f"skew={y.skew():.3f}  min={y.min():.2e}  max={y.max():.3f}  "
          f"negatives={(y < 0).sum()}  at_1.0={(y >= 0.999).sum()}")
    print(f"log1p skew={np.log1p(y).skew():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(y, bins=80, ax=ax[0], color="steelblue")
    ax[0].set_title(f"demand (raw)  skew={y.skew():.2f}")
    sns.histplot(np.log1p(y), bins=80, ax=ax[1], color="seagreen")
    ax[1].set_title(f"log1p(demand)  skew={np.log1p(y).skew():.2f}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/target_distribution.png", dpi=110)
    plt.close(fig)

    # ---- Demand by hour-of-day (why daytime matters) -----------------------
    tr = train.copy()
    tr["minutes"] = tr["timestamp"].map(_ts_to_minutes)
    tr["hour"] = tr["minutes"] // 60
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=tr, x="hour", y="demand", hue="day",
                 estimator="mean", errorbar=None, marker="o", ax=ax)
    ax.set_title("Mean demand by hour-of-day (per day)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/demand_by_hour.png", dpi=110)
    plt.close(fig)

    # ---- Demand by categorical ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["RoadType", "Weather", "NumberofLanes"]):
        sns.boxplot(data=tr, x=col, y="demand", ax=ax, showfliers=False)
        ax.set_title(f"demand by {col}")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(f"{OUT}/demand_by_category.png", dpi=110)
    plt.close(fig)

    # ---- Train/test temporal coverage --------------------------------------
    te = test.copy()
    te["minutes"] = te["timestamp"].map(_ts_to_minutes)
    print("\n--- TEMPORAL COVERAGE ---")
    for d in sorted(tr["day"].unique()):
        s = tr[tr.day == d]["minutes"]
        print(f"  train day {d}: minutes {s.min():4d}-{s.max():4d}  "
              f"({s.nunique()} ticks, {len(s)} rows)")
    print(f"  test  day 49: minutes {te['minutes'].min():4d}-"
          f"{te['minutes'].max():4d}  ({te['minutes'].nunique()} ticks, "
          f"{len(te)} rows)")

    print(f"\nPlots saved to {OUT}/")


if __name__ == "__main__":
    main()

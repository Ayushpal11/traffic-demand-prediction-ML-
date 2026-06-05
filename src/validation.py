"""
Validation strategy & competition metric.

The task is a spatio-temporal forecast: test = day 49 daytime (2:15-13:45),
which never appears in the train timestamps. To avoid leakage and to align the
local score with the leaderboard we provide two complementary tools:

1. GroupKFold by geohash  -> stable OOF for every train row (used for
   ensembling / target encoding). Generalises across locations.
2. time_holdout_mask      -> a leaderboard-mimicking split: hold out day-48
   rows whose minute-of-day falls in the test window. This mirrors the real
   task (predict daytime from the rest of the day) and is the score we trust
   most for ranking models.
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


# --- Competition metric -----------------------------------------------------
def competition_score(y_true, y_pred):
    """Score = max(0, 100 * R^2)."""
    return max(0.0, 100.0 * r2_score(y_true, y_pred))


# --- GroupKFold by geohash --------------------------------------------------
def make_group_folds(df, group_col="geohash", n_splits=5):
    """Return an int array fold_id in [0, n_splits) aligned to df rows."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_ids = np.full(len(df), -1, dtype=int)
    groups = df[group_col].values
    X_dummy = np.zeros(len(df))
    for f, (_, val_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        fold_ids[val_idx] = f
    assert (fold_ids >= 0).all()
    return fold_ids


# --- Test-mimicking time holdout --------------------------------------------
# Test covers minutes 135 (2:15) .. 825 (13:45) on day 49.
TEST_MIN_LO, TEST_MIN_HI = 135, 825


def time_holdout_mask(df):
    """
    Boolean mask of rows used as the time-holdout VALIDATION set:
    day-48 rows whose minute-of-day lies inside the test daytime window.
    Train on ~df[~mask], validate on df[mask].
    """
    mins = df["minutes_of_day"].values
    return (df["day"].values == 48) & (mins >= TEST_MIN_LO) & (mins <= TEST_MIN_HI)

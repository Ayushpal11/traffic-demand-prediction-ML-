"""
STEP 6 — Blending & final submission.

- Loads OOF + test predictions for LGB / XGB / CatBoost.
- Finds weighted-average blend weights that maximise the OOF competition score
  (constrained to the simplex), and also reports the test-mimicking time-holdout
  score so we trust the choice.
- Falls back to the prompt's suggested 0.5/0.3/0.2 weights for reference.
- Post-processes (clip to [0, 1]) and writes submission.csv (41778 x 2),
  columns exactly matching sample_submission.csv.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.validation import competition_score

OOF_DIR = "outputs/oof"
MODELS = ["lgb", "xgb", "cat"]


def load():
    y_raw = np.load(f"{OOF_DIR}/y_raw.npy")
    th_mask = np.load(f"{OOF_DIR}/th_mask.npy")
    oof = {m: np.load(f"{OOF_DIR}/oof_{m}.npy") for m in MODELS}
    test = {m: np.load(f"{OOF_DIR}/test_{m}.npy") for m in MODELS}
    return y_raw, th_mask, oof, test


def blend(weights, preds):
    w = np.asarray(weights)
    return sum(w[i] * preds[m] for i, m in enumerate(MODELS))


def optimize_weights(oof, y_raw):
    """Maximise OOF score over the weight simplex (SLSQP, multi-start)."""
    def neg_score(w):
        p = np.clip(blend(w, oof), 0, 1)
        return -competition_score(y_raw, p)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * len(MODELS)
    best = None
    for start in [np.full(3, 1 / 3), [0.5, 0.2, 0.3], [0.6, 0.2, 0.2]]:
        res = minimize(neg_score, start, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def main():
    y_raw, th_mask, oof, test = load()

    print("=== Individual OOF scores ===")
    for m in MODELS:
        full = competition_score(y_raw, np.clip(oof[m], 0, 1))
        th = competition_score(y_raw[th_mask], np.clip(oof[m][th_mask], 0, 1))
        print(f"  {m}: OOF={full:.4f}  time-holdout={th:.4f}")

    # ---- Reference blend (prompt's suggested weights) ----------------------
    ref_w = [0.5, 0.2, 0.3]  # lgb, xgb, cat  (0.5 LGB / 0.3 CAT / 0.2 XGB)
    ref = np.clip(blend(ref_w, oof), 0, 1)
    print(f"\nReference blend {dict(zip(MODELS, ref_w))}: "
          f"OOF={competition_score(y_raw, ref):.4f}  "
          f"time-holdout={competition_score(y_raw[th_mask], ref[th_mask]):.4f}")

    # ---- Optimised blend ----------------------------------------------------
    w = optimize_weights(oof, y_raw)
    opt = np.clip(blend(w, oof), 0, 1)
    print(f"\nOptimised weights {dict(zip(MODELS, np.round(w, 3)))}:")
    print(f"  OOF={competition_score(y_raw, opt):.4f}  "
          f"time-holdout={competition_score(y_raw[th_mask], opt[th_mask]):.4f}")

    # ---- Pick the better of the two on the time-holdout (test-aligned) ------
    score_ref = competition_score(y_raw[th_mask], ref[th_mask])
    score_opt = competition_score(y_raw[th_mask], opt[th_mask])
    if score_opt >= score_ref:
        final_w, tag = w, "optimised"
    else:
        final_w, tag = ref_w, "reference"
    print(f"\nChosen blend: {tag}  weights={dict(zip(MODELS, np.round(final_w, 3)))}")

    # ---- Build submission ---------------------------------------------------
    test_pred = np.clip(blend(final_w, test), 0, 1)  # post-process: no negatives
    sample = pd.read_csv("dataset/sample_submission.csv")
    test_raw = pd.read_csv("dataset/test.csv")
    sub = pd.DataFrame({"Index": test_raw["Index"].values, "demand": test_pred})
    # match sample column order/names exactly
    sub = sub[list(sample.columns)]

    assert sub.shape == (41778, 2), f"bad shape {sub.shape}"
    assert list(sub.columns) == list(sample.columns), "column mismatch"
    assert (sub["demand"] >= 0).all(), "negative predictions remain"
    sub.to_csv("submission.csv", index=False)
    print(f"\nsubmission.csv written: shape={sub.shape}, "
          f"cols={list(sub.columns)}")
    print(sub.head())
    print(f"pred range: [{test_pred.min():.4f}, {test_pred.max():.4f}]  "
          f"mean={test_pred.mean():.4f}")


if __name__ == "__main__":
    main()

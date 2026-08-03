"""Which distribution should turn a projected count into an over/under
probability?

The deployed answer was a Normal centered on the model's projection with one
pooled residual std. Real graded predictions on the live site say that runs
the batting props badly hot -- Hits stated 48.7% over against a real 32.5%,
RBI 45.7% against 26.1%, across ~9,300 real graded predictions. Two suspects,
both structural rather than a tuning problem: these are discrete counts (an
integer line of 1 means "2 or more", and a Normal counts the impossible mass
between 1 and 2), and their real distribution is right-skewed with a large
point mass at zero rather than symmetric.

This compares four candidates on a real, untouched holdout, walk-forward
(refit per calendar date on strictly earlier games only, same discipline as
backtest_props.py):

  normal          - deployed: 1 - Phi(line)
  normal_cc       - same, with a discreteness correction: 1 - Phi(floor(line)+0.5)
  poisson         - 1 - PoissonCDF(floor(line)), variance forced to equal mean
  negbin          - 1 - NegBinCDF(floor(line)), dispersion fit on train only

Scored by Brier (does the stated probability match reality), log loss, and
the calibration gap -- mean stated probability minus the rate the event
really happened, which is the exact quantity the site's calibration chart
plots and the one the Normal was getting wrong.

Lines mirror how the live site sets and grades them: anchored to the model's
own projection rounded to the nearest half (project_count_stat), graded as
actual > line.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.props.prop_data import build_batter_prop_table, build_pitcher_prop_table
from pipeline.mlb.props.prop_models import (
    FEATURES, over_prob, count_over_prob, estimate_dispersion,
)
from pipeline.common.metrics import brier_score, log_loss
from sklearn.linear_model import RidgeCV
from scipy.stats import norm

TEST_START = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2025-09-30")

# Every count prop the live MLB slate actually ships, batting and pitching --
# the pitching ones are included deliberately as a control: if the diagnosis
# (Normal breaks down on SMALL counts) is right, they should show a much
# smaller gap than the batting ones, since their counts are far larger.
BATTER_STATS = ["hits", "total_bases", "rbi", "walks"]
PITCHER_STATS = ["strikeouts", "hits_allowed", "walks_allowed", "runs_allowed", "outs_recorded"]


def walk_forward_with_dispersion(df, test_dates, min_train=200):
    """Same walk-forward fit as prop_models.walk_forward_count_stat, but also
    returns the negative-binomial dispersion estimated on each date's own
    training slice -- fit on train only, never on the held-out rows."""
    n = len(df)
    pred = np.full(n, np.nan)
    std = np.full(n, np.nan)
    disp = np.full(n, np.nan)

    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < min_train or len(target_idx) == 0:
            continue
        X_train, y_train = train[FEATURES].values, train["actual"].values
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)

        pos = df.index.get_indexer(target_idx)
        pred[pos] = model.predict(df.loc[target_idx, FEATURES].values)
        std[pos] = max(float(np.std(y_train - train_pred)), 1e-6)
        disp[pos] = estimate_dispersion(y_train, train_pred)

    return pred, std, disp


def evaluate(df, label, out):
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    if not test_dates:
        print(f"  {label}: no test rows, skipped")
        return
    pred, std, disp = walk_forward_with_dispersion(df, test_dates)

    valid = df["game_date"].isin(test_dates).values & ~np.isnan(pred)
    if valid.sum() < 100:
        print(f"  {label}: only {valid.sum()} usable rows, skipped")
        return

    actual = df["actual"].values[valid]
    mu, sd, al = pred[valid], std[valid], disp[valid]

    # Real line-setting and grading, copied from the live generator.
    line = np.round(mu * 2) / 2.0
    line = np.where(line <= 0, 0.5, line)
    over = (actual > line).astype(float)

    mean_alpha = float(np.nanmean(al))
    candidates = {
        "normal": over_prob(mu, sd, line),
        "normal_cc": 1.0 - norm.cdf(np.floor(line) + 0.5, loc=mu, scale=sd),
        "poisson": count_over_prob(mu, line, 0.0),
        "negbin": count_over_prob(mu, line, mean_alpha),
    }

    actual_rate = float(over.mean())
    print(f"\n--- {label} (n={int(valid.sum())}, real over-rate {actual_rate:.3f}, fitted alpha {mean_alpha:.3f}) ---")
    print(f"    {'model':<11} {'says':>7} {'gap':>8} {'brier':>8} {'logloss':>9}")
    rows = {}
    for name, p in candidates.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        says = float(p.mean())
        rows[name] = {
            "mean_predicted": says, "gap": says - actual_rate,
            "brier": brier_score(over, p), "log_loss": log_loss(over, p),
        }
        print(f"    {name:<11} {says:>7.3f} {says-actual_rate:>+8.3f} "
              f"{rows[name]['brier']:>8.4f} {rows[name]['log_loss']:>9.4f}")

    best = min(rows, key=lambda k: rows[k]["brier"])
    print(f"    best by Brier: {best}")
    out[label] = {"n": int(valid.sum()), "actual_rate": actual_rate,
                  "mean_alpha": mean_alpha, "candidates": rows, "best_by_brier": best}


def main():
    print("=== MLB count-prop distribution backtest ===")
    print(f"Walk-forward, holdout {TEST_START.date()} to {TEST_END.date()}\n")
    out = {}

    for stat in BATTER_STATS:
        evaluate(build_batter_prop_table(stat), f"batter_{stat}", out)
    for stat in PITCHER_STATS:
        evaluate(build_pitcher_prop_table(stat), f"pitcher_{stat}", out)

    print("\n=== summary: wins by Brier ===")
    tally = {}
    for label, r in out.items():
        tally[r["best_by_brier"]] = tally.get(r["best_by_brier"], 0) + 1
        print(f"  {label:<26} {r['best_by_brier']:<10} "
              f"(normal gap {r['candidates']['normal']['gap']:+.3f} -> "
              f"{r['best_by_brier']} gap {r['candidates'][r['best_by_brier']]['gap']:+.3f})")
    print(f"\n  totals: {tally}")

    path = ROOT / "notebooks_out" / "mlb_count_distribution_backtest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

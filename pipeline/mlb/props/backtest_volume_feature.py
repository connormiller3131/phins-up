"""Does the prop model improve when it can see playing time and quality of
contact, not just the blended per-game rate?

A counting stat is rate x opportunities. The deployed feature set
[own_trailing_avg, opp_allowed_trailing_avg] only ever sees the blended
per-game production rate, so identical recent totals from a 4.5-PA-per-game
leadoff hitter and a 2-PA bench bat look the same. Candidates tested here,
walk-forward on the same untouched Aug-Sep 2025 holdout as every other prop
backtest (refit per calendar date, train strictly earlier):

  baseline       [own_trailing_avg, opp_allowed_trailing_avg]   (deployed)
  +volume        + own_trailing_volume  (trailing pa_count / batters_faced)
  +xwoba         + own_trailing_est_woba (batters only -- Statcast expected
                   wOBA, quality of contact with batted-ball luck stripped)
  +volume+xwoba  both (batters only)

Design notes, so the comparison is honest:
- One table per stat containing every candidate feature, all candidates
  evaluated on that SAME row set. est_woba is NaN on ~11% of raw games
  (no batted ball), which shrinks the shared table slightly vs deployed;
  the within-table comparison is what decides.
- RMSE/MAE compare projection means, line-free.
- Brier/log-loss/calibration-gap use the deployed negative-binomial
  machinery (count_over_prob, dispersion fit on train only) against a
  COMMON neutral line per row -- the naive own_trailing_avg rounded to the
  nearest half, identical for every candidate -- so probability scores are
  on the same events. Grading a candidate on its own self-anchored line
  would give each candidate a different event set and make Brier
  incomparable.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.props.prop_data import build_batter_prop_table, build_pitcher_prop_table
from pipeline.mlb.props.prop_models import count_over_prob, estimate_dispersion
from pipeline.common.metrics import brier_score, log_loss
from sklearn.linear_model import RidgeCV

TEST_START = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2025-09-30")

BASELINE = ["own_trailing_avg", "opp_allowed_trailing_avg"]

BATTER_STATS = ["hits", "total_bases", "rbi", "walks"]
PITCHER_STATS = ["strikeouts", "hits_allowed", "walks_allowed", "runs_allowed", "outs_recorded"]

BATTER_CANDIDATES = {
    "baseline": BASELINE,
    "+volume": BASELINE + ["own_trailing_volume"],
    "+xwoba": BASELINE + ["own_trailing_est_woba"],
    "+volume+xwoba": BASELINE + ["own_trailing_volume", "own_trailing_est_woba"],
}
PITCHER_CANDIDATES = {
    "baseline": BASELINE,
    "+volume": BASELINE + ["own_trailing_volume"],
}


def walk_forward(df, features, test_dates, min_train=200):
    n = len(df)
    pred = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < min_train or len(target_idx) == 0:
            continue
        X_train, y_train = train[features].values, train["actual"].values
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(X_train, y_train)
        pos = df.index.get_indexer(target_idx)
        pred[pos] = model.predict(df.loc[target_idx, features].values)
        disp[pos] = estimate_dispersion(y_train, model.predict(X_train))
    return pred, disp


def evaluate(df, candidates, label, out):
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    if not test_dates:
        print(f"  {label}: no test rows, skipped")
        return

    runs = {name: walk_forward(df, feats, test_dates) for name, feats in candidates.items()}

    # Shared validity mask: every candidate must have a prediction for the
    # row, so all scores are on the identical event set.
    valid = df["game_date"].isin(test_dates).values
    for pred, _ in runs.values():
        valid &= ~np.isnan(pred)
    if valid.sum() < 100:
        print(f"  {label}: only {valid.sum()} usable rows, skipped")
        return

    actual = df["actual"].values[valid]
    naive = df["own_trailing_avg"].values[valid]
    common_line = np.round(naive * 2) / 2.0
    common_line = np.where(common_line <= 0, 0.5, common_line)
    over = (actual > common_line).astype(float)
    actual_rate = float(over.mean())

    print(f"\n--- {label} (n={int(valid.sum())}, common-line over-rate {actual_rate:.3f}) ---")
    print(f"    {'candidate':<15} {'rmse':>7} {'mae':>7} {'brier':>8} {'logloss':>9} {'gap':>8}")
    rows = {}
    for name in candidates:
        pred, disp = runs[name]
        mu, al = pred[valid], disp[valid]
        p = np.clip(count_over_prob(mu, common_line, float(np.nanmean(al))), 1e-6, 1 - 1e-6)
        rows[name] = {
            "rmse": float(np.sqrt(np.mean((actual - mu) ** 2))),
            "mae": float(np.mean(np.abs(actual - mu))),
            "brier": brier_score(over, p),
            "log_loss": log_loss(over, p),
            "gap": float(p.mean()) - actual_rate,
        }
        r = rows[name]
        print(f"    {name:<15} {r['rmse']:>7.4f} {r['mae']:>7.4f} {r['brier']:>8.4f} {r['log_loss']:>9.4f} {r['gap']:>+8.3f}")

    best = min(rows, key=lambda k: rows[k]["brier"])
    d_brier = rows["baseline"]["brier"] - rows[best]["brier"]
    d_rmse = rows["baseline"]["rmse"] - min(rows.values(), key=lambda r: r["rmse"])["rmse"]
    print(f"    best by Brier: {best} (beats baseline by {d_brier:.4f}); best RMSE beats baseline by {d_rmse:.4f}")
    out[label] = {"n": int(valid.sum()), "actual_rate": actual_rate,
                  "candidates": rows, "best_by_brier": best}


def main():
    print("=== MLB prop volume/xwOBA feature backtest ===")
    print(f"Walk-forward, holdout {TEST_START.date()} to {TEST_END.date()}\n")
    out = {}

    for stat in BATTER_STATS:
        df = build_batter_prop_table(stat, with_volume=True, extra_trailing_cols=("est_woba",))
        evaluate(df, BATTER_CANDIDATES, f"batter_{stat}", out)

    for stat in PITCHER_STATS:
        df = build_pitcher_prop_table(stat, with_volume=True)
        evaluate(df, PITCHER_CANDIDATES, f"pitcher_{stat}", out)

    print("\n=== summary ===")
    for label, r in out.items():
        base, best = r["candidates"]["baseline"], r["candidates"][r["best_by_brier"]]
        print(f"  {label:<26} best={r['best_by_brier']:<15} "
              f"brier {base['brier']:.4f} -> {best['brier']:.4f}  "
              f"rmse {base['rmse']:.4f} -> {best['rmse']:.4f}")

    path = ROOT / "notebooks_out" / "mlb_volume_feature_backtest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

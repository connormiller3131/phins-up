"""Backtest NFL player-prop models, walk-forward, out-of-sample on 2023-2024
(train/refit only ever sees strictly earlier games).

No free source of real historical player-prop odds exists, so two honest,
non-circular evaluations are used instead of comparing to a market line:

1. Yardage props: RMSE/MAE of the opponent-adjusted projection vs. actual
   yards, compared against a naive baseline (player's own trailing average,
   ignoring the opponent entirely). This needs no invented betting line.
2. Anytime-TD: this is inherently a probability of a binary event, so
   Brier score / log loss / calibration are computed directly against actual
   outcomes -- also no line needed. Compared against a naive baseline that
   uses only the player's own trailing TD rate (no opponent adjustment).

For illustration only, yardage props also report Brier/log loss using the
player's own trailing average AS A PROXY sportsbook line (a common rule of
thumb -- real lines track recent form closely). That number should be read
as "does the opponent adjustment move probability in the right direction
relative to a same-line naive coin flip", not as a real market backtest.
"""
import sys
import pathlib
import json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.props.prop_data import build_prop_table
from pipeline.nfl.props.prop_models import walk_forward_yardage, walk_forward_anytime_td, yardage_over_prob
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

TEST_SEASONS = [2025]  # freshest completed season, never touched during development

YARDAGE_PROPS = {
    "passing_yards": ["QB"],
    "rushing_yards": ["RB"],
    "receiving_yards": ["WR", "TE"],
}


def backtest_yardage(stat_col, positions):
    df = build_prop_table(stat_col, positions)
    model_pred, resid_std, naive_pred = walk_forward_yardage(df, TEST_SEASONS)

    test_mask = df["season"].isin(TEST_SEASONS)
    valid = test_mask.values & ~np.isnan(model_pred)
    actual = df["actual"].values[valid]
    mp, rs, np_ = model_pred[valid], resid_std[valid], naive_pred[valid]

    rmse_model = float(np.sqrt(np.mean((actual - mp) ** 2)))
    mae_model = float(np.mean(np.abs(actual - mp)))
    rmse_naive = float(np.sqrt(np.mean((actual - np_) ** 2)))
    mae_naive = float(np.mean(np.abs(actual - np_)))

    # proxy-line illustration (line = naive own-average, rounded like a book would)
    line = np.round(np_ * 2) / 2.0
    over_actual = (actual > line).astype(float)
    model_p_over = yardage_over_prob(mp, rs, line)
    naive_p_over = np.full_like(model_p_over, 0.5)

    result = {
        "n": int(valid.sum()),
        "rmse_model": rmse_model, "mae_model": mae_model,
        "rmse_naive": rmse_naive, "mae_naive": mae_naive,
        "proxy_line_brier_model": brier_score(over_actual, model_p_over),
        "proxy_line_brier_naive": brier_score(over_actual, naive_p_over),
        "proxy_line_logloss_model": log_loss(over_actual, model_p_over),
        "proxy_line_logloss_naive": log_loss(over_actual, naive_p_over),
        "proxy_line_calibration": calibration_curve(over_actual, model_p_over, n_bins=10),
    }
    return result


def backtest_anytime_td():
    df = build_prop_table("anytime_td", ["RB", "WR", "TE"])
    model_pred = walk_forward_anytime_td(df, TEST_SEASONS)

    test_mask = df["season"].isin(TEST_SEASONS)
    valid = test_mask.values & ~np.isnan(model_pred)
    actual = df["actual"].values[valid]
    mp = model_pred[valid]
    naive_pred = df["own_trailing_avg"].values[valid]  # own trailing TD rate, no opponent info

    result = {
        "n": int(valid.sum()),
        "model_brier": brier_score(actual, mp),
        "model_logloss": log_loss(actual, mp),
        "model_accuracy": accuracy(actual, mp),
        "naive_brier": brier_score(actual, naive_pred),
        "naive_logloss": log_loss(actual, naive_pred),
        "naive_accuracy": accuracy(actual, naive_pred),
        "model_calibration": calibration_curve(actual, mp, n_bins=10),
    }
    return result


def main():
    all_results = {}

    print("=== Yardage props (opponent-adjusted ridge vs. naive own-average) ===")
    for stat_col, positions in YARDAGE_PROPS.items():
        print(f"\n--- {stat_col} ({'/'.join(positions)}) ---")
        r = backtest_yardage(stat_col, positions)
        all_results[stat_col] = r
        print(f"  n={r['n']}")
        print(f"  RMSE  model={r['rmse_model']:.2f}  naive={r['rmse_naive']:.2f}")
        print(f"  MAE   model={r['mae_model']:.2f}  naive={r['mae_naive']:.2f}")
        print(f"  [proxy-line illustration] Brier model={r['proxy_line_brier_model']:.4f} naive(0.5)={r['proxy_line_brier_naive']:.4f}")
        print(f"  [proxy-line illustration] LogLoss model={r['proxy_line_logloss_model']:.4f} naive(0.5)={r['proxy_line_logloss_naive']:.4f}")

    print("\n=== Anytime TD (logistic, opponent-adjusted vs. naive own-rate) ===")
    td = backtest_anytime_td()
    all_results["anytime_td"] = td
    print(f"  n={td['n']}")
    print(f"  Brier   model={td['model_brier']:.4f}  naive={td['naive_brier']:.4f}")
    print(f"  LogLoss model={td['model_logloss']:.4f}  naive={td['naive_logloss']:.4f}")
    print(f"  Accuracy model={td['model_accuracy']:.4f}  naive={td['naive_accuracy']:.4f}")
    print("  Calibration (predicted -> actual, count):")
    for b in td["model_calibration"]:
        print(f"    [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}]  pred={b['predicted_mean']:.3f}  actual={b['actual_mean']:.3f}  n={b['count']}")

    out_path = ROOT / "notebooks_out" / "nfl_props_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()

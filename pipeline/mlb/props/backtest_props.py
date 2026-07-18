"""Backtest MLB player-prop models, walk-forward (refit once per calendar
date, using only strictly earlier games). Test window: August-September 2025
(train = everything before that, i.e. all of 2024 plus early 2025) -- a real
holdout with a large enough sample without refitting for every single day
of a full season.

Same honesty caveats as NFL: no free source of real historical player-prop
odds exists, so count props are evaluated primarily via RMSE/MAE against a
naive (own-trailing-average-only) baseline, plus a proxy-line illustration.
Anytime-HR is inherently a probability and is scored directly.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.props.prop_data import build_batter_prop_table, build_pitcher_prop_table
from pipeline.mlb.props.prop_models import walk_forward_count_stat, walk_forward_binary_stat, over_prob
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

TEST_START = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2025-09-30")

COUNT_PROPS = {
    "hits": ("batter", None),
    "total_bases": ("batter", None),
    "pitcher_strikeouts": ("pitcher", None),
}


def backtest_count_stat(df, label):
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    model_pred, resid_std, naive_pred = walk_forward_count_stat(df, test_dates)

    test_mask = df["game_date"].isin(test_dates)
    valid = test_mask.values & ~np.isnan(model_pred)
    actual = df["actual"].values[valid]
    mp, rs, np_ = model_pred[valid], resid_std[valid], naive_pred[valid]

    rmse_model = float(np.sqrt(np.mean((actual - mp) ** 2)))
    mae_model = float(np.mean(np.abs(actual - mp)))
    rmse_naive = float(np.sqrt(np.mean((actual - np_) ** 2)))
    mae_naive = float(np.mean(np.abs(actual - np_)))

    line = np.round(np_ * 2) / 2.0
    over_actual = (actual > line).astype(float)
    model_p_over = over_prob(mp, rs, line)
    naive_p_over = np.full_like(model_p_over, 0.5)

    result = {
        "n": int(valid.sum()),
        "rmse_model": rmse_model, "mae_model": mae_model,
        "rmse_naive": rmse_naive, "mae_naive": mae_naive,
        "proxy_line_brier_model": brier_score(over_actual, model_p_over),
        "proxy_line_brier_naive": brier_score(over_actual, naive_p_over),
        "proxy_line_logloss_model": log_loss(over_actual, model_p_over),
        "proxy_line_logloss_naive": log_loss(over_actual, naive_p_over),
    }
    print(f"\n--- {label} ---")
    print(f"  n={result['n']}")
    print(f"  RMSE  model={result['rmse_model']:.3f}  naive={result['rmse_naive']:.3f}")
    print(f"  MAE   model={result['mae_model']:.3f}  naive={result['mae_naive']:.3f}")
    print(f"  [proxy-line] Brier model={result['proxy_line_brier_model']:.4f} naive={result['proxy_line_brier_naive']:.4f}")
    return result


def backtest_anytime_hr():
    df = build_batter_prop_table("home_runs")
    df["actual"] = (df["actual"] > 0).astype(float)
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    model_pred = walk_forward_binary_stat(df, test_dates)

    test_mask = df["game_date"].isin(test_dates)
    valid = test_mask.values & ~np.isnan(model_pred)
    actual = df["actual"].values[valid]
    mp = model_pred[valid]
    naive_pred = df["own_trailing_avg"].values[valid]

    result = {
        "n": int(valid.sum()),
        "model_brier": brier_score(actual, mp), "naive_brier": brier_score(actual, naive_pred),
        "model_logloss": log_loss(actual, mp), "naive_logloss": log_loss(actual, naive_pred),
        "model_accuracy": accuracy(actual, mp), "naive_accuracy": accuracy(actual, naive_pred),
        "calibration": calibration_curve(actual, mp, n_bins=10),
    }
    print("\n--- anytime_home_run ---")
    print(f"  n={result['n']}")
    print(f"  Brier   model={result['model_brier']:.4f}  naive={result['naive_brier']:.4f}")
    print(f"  LogLoss model={result['model_logloss']:.4f}  naive={result['naive_logloss']:.4f}")
    print(f"  Accuracy model={result['model_accuracy']:.4f}  naive={result['naive_accuracy']:.4f}")
    return result


def main():
    all_results = {}

    print("=== MLB player props backtest, test window 2025-08-01 to 2025-09-30 ===")
    hits_df = build_batter_prop_table("hits")
    all_results["hits"] = backtest_count_stat(hits_df, "hits (batter)")

    tb_df = build_batter_prop_table("total_bases")
    all_results["total_bases"] = backtest_count_stat(tb_df, "total_bases (batter)")

    pk_df = build_pitcher_prop_table()
    all_results["pitcher_strikeouts"] = backtest_count_stat(pk_df, "pitcher_strikeouts")

    all_results["anytime_home_run"] = backtest_anytime_hr()

    out_path = ROOT / "notebooks_out" / "mlb_props_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

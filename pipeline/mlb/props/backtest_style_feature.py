"""Does adding batter-vs-pitching-style (fastball/sinker/slider/curve/etc
cluster, not just handedness) improve the prop models over the current
[own_trailing_avg, opp_allowed_trailing_avg] baseline? Same rigor, same
2025-08-01 to 2025-09-30 walk-forward test window as every other prop
backtest in this project.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.props.prop_data import build_batter_prop_table
from pipeline.mlb.props.style_features import attach_style_feature

TEST_START = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2025-09-30")


def walk_forward_compare(df, features):
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    n = len(df)
    preds = np.full(n, np.nan)
    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < 200:
            continue
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(train[features].values, train["actual"].values)
        pos = df.index.get_indexer(target_idx)
        preds[pos] = model.predict(df.loc[target_idx, features].values)
    return preds


def main():
    for stat_col in ["hits", "total_bases", "home_runs"]:
        base = build_batter_prop_table(stat_col)
        with_style = attach_style_feature(base.copy(), stat_col)
        with_style = with_style.dropna(subset=["own_vs_style_trailing_avg"]).reset_index(drop=True)

        base_test_mask = base["game_date"].between(TEST_START, TEST_END)
        style_test_mask = with_style["game_date"].between(TEST_START, TEST_END)

        base_preds = walk_forward_compare(base, ["own_trailing_avg", "opp_allowed_trailing_avg"])
        style_preds = walk_forward_compare(
            with_style, ["own_trailing_avg", "opp_allowed_trailing_avg", "own_vs_style_trailing_avg"])

        b_valid = base_test_mask.values & ~np.isnan(base_preds)
        s_valid = style_test_mask.values & ~np.isnan(style_preds)

        b_actual, b_pred = base["actual"].values[b_valid], base_preds[b_valid]
        s_actual, s_pred = with_style["actual"].values[s_valid], style_preds[s_valid]

        print(f"\n--- {stat_col} ---")
        print(f"  baseline (own+opp):          n={b_valid.sum():5d}  RMSE={np.sqrt(np.mean((b_actual-b_pred)**2)):.4f}  MAE={np.mean(np.abs(b_actual-b_pred)):.4f}")
        print(f"  +vs_style (own+opp+style):   n={s_valid.sum():5d}  RMSE={np.sqrt(np.mean((s_actual-s_pred)**2)):.4f}  MAE={np.mean(np.abs(s_actual-s_pred)):.4f}")


if __name__ == "__main__":
    main()

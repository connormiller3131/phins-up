"""Does adding the batter-vs-pitcher-handedness trailing rate actually
improve the batter prop models, or is the team-wide opponent-allowed rate
already good enough? Compares [own_trailing_avg, opp_allowed_trailing_avg]
(current) against the same plus own_vs_hand_trailing_avg, on the identical
2025-08-01 to 2025-09-30 walk-forward test window used everywhere else.
Only deploy if this is a real improvement -- same rule as every other model
change in this project.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.props.prop_data import build_batter_prop_table
from pipeline.mlb.props.hand_features import attach_vs_hand_feature

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
    for stat_col, positions in [("hits", None), ("total_bases", None), ("home_runs", None)]:
        base = build_batter_prop_table(stat_col)
        with_hand = attach_vs_hand_feature(base.copy(), stat_col)
        with_hand = with_hand.dropna(subset=["own_vs_hand_trailing_avg"]).reset_index(drop=True)

        base_test_mask = base["game_date"].between(TEST_START, TEST_END)
        hand_test_mask = with_hand["game_date"].between(TEST_START, TEST_END)

        base_preds = walk_forward_compare(base, ["own_trailing_avg", "opp_allowed_trailing_avg"])
        hand_preds = walk_forward_compare(
            with_hand, ["own_trailing_avg", "opp_allowed_trailing_avg", "own_vs_hand_trailing_avg"])

        b_valid = base_test_mask.values & ~np.isnan(base_preds)
        h_valid = hand_test_mask.values & ~np.isnan(hand_preds)

        b_actual, b_pred = base["actual"].values[b_valid], base_preds[b_valid]
        h_actual, h_pred = with_hand["actual"].values[h_valid], hand_preds[h_valid]

        print(f"\n--- {stat_col} ---")
        print(f"  baseline (own+opp):        n={b_valid.sum():5d}  RMSE={np.sqrt(np.mean((b_actual-b_pred)**2)):.4f}  MAE={np.mean(np.abs(b_actual-b_pred)):.4f}")
        print(f"  +vs_hand (own+opp+hand):   n={h_valid.sum():5d}  RMSE={np.sqrt(np.mean((h_actual-h_pred)**2)):.4f}  MAE={np.mean(np.abs(h_actual-h_pred)):.4f}")
        print(f"  (n differs: vs_hand requires {20}-game same-handedness trailing history, drops more rows)")


if __name__ == "__main__":
    main()

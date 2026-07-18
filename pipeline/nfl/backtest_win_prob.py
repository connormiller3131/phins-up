"""Backtest NFL win-probability models: Elo (grid-searched hyperparams) and
ridge-regression margin ratings, both evaluated out-of-sample against a
no-vig market-implied-probability baseline built from the moneylines already
present in the nflverse schedules data.

Train (hyperparam fitting only): 2019-2024
Held-out test (all reported metrics): 2025 (most recently completed season)
"""
import sys
import pathlib
import json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.games import load_games
from pipeline.nfl.elo_model import run_elo, fit_elo_hyperparams
from pipeline.nfl.ridge_margin_model import walk_forward_ridge
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_SEASONS = [2025]


def main():
    df = load_games()
    print(f"Loaded {len(df)} completed games, seasons {df['season'].min()}-{df['season'].max()}")

    train_df = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test_mask = df["season"].isin(TEST_SEASONS)
    n_ties = int((df.loc[test_mask, "home_win"] == 0.5).sum())
    print(f"Test set: {test_mask.sum()} games ({n_ties} ties excluded from win/loss metrics)")

    # ---- Elo: fit hyperparams on train seasons only ----
    print("\nFitting Elo hyperparameters (K, home_adv, scale, rest_adv, season_regression) on 2019-2024 via grid search...")
    elo_params = fit_elo_hyperparams(train_df)
    print("Best Elo params:", elo_params)

    # Run Elo across the FULL chronological sequence so test-season ratings
    # carry forward correctly, then slice out test predictions.
    elo_preds_full = run_elo(df, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"],
                             rest_adv=elo_params["rest_adv"], season_regression=elo_params["season_regression"])
    elo_test_preds = elo_preds_full[test_mask.values]

    # ---- Ridge margin: walk-forward, refit before every test week ----
    print(f"\nRunning walk-forward ridge margin model over {TEST_SEASONS} (refits every week)...")
    ridge_preds_full = walk_forward_ridge(df, TEST_SEASONS)
    ridge_test_preds = ridge_preds_full[test_mask.values]

    # ---- Market baseline (no-vig moneyline implied probability) ----
    market_test_preds = df.loc[test_mask, "market_home_prob"].values

    y_test = df.loc[test_mask, "home_win"].values
    non_tie = y_test != 0.5

    results = {}
    for name, preds in [("elo", elo_test_preds), ("ridge_margin", ridge_test_preds), ("market", market_test_preds)]:
        valid = non_tie & ~np.isnan(preds)
        results[name] = {
            "n_games": int(valid.sum()),
            "brier": brier_score(y_test[valid], preds[valid]),
            "log_loss": log_loss(y_test[valid], preds[valid]),
            "accuracy": accuracy(y_test[valid], preds[valid]),
            "calibration": calibration_curve(y_test[valid], preds[valid], n_bins=10),
        }

    print(f"\n=== Out-of-sample results, {TEST_SEASONS} test season(s) ===")
    for name, r in results.items():
        print(f"\n{name}  (n={r['n_games']})")
        print(f"  Brier score: {r['brier']:.4f}")
        print(f"  Log loss:    {r['log_loss']:.4f}")
        print(f"  Accuracy:    {r['accuracy']:.4f}")
        print("  Calibration (predicted -> actual, count):")
        for b in r["calibration"]:
            print(f"    [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}]  pred={b['predicted_mean']:.3f}  actual={b['actual_mean']:.3f}  n={b['count']}")

    out_path = ROOT / "notebooks_out" / "nfl_win_prob_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"elo_params": elo_params, "results": results}, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()

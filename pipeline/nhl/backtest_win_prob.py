"""Backtest NHL win-probability Elo model.

Train (hyperparameter fitting only): 2018-19 through 2023-24 -- six seasons,
including the COVID-truncated 2019-20 and the delayed, shortened, realigned
2020-21. Held-out test: the entire 2024-25 season, completely unseen during
fitting. Ratings carry forward chronologically through the full history
into the test season, exactly as they do in live use.
"""
import sys
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nhl.games import load_games
from pipeline.nhl.elo_model import run_elo, fit_elo_hyperparams
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
TEST_SEASONS = [2024]


def main():
    df = load_games()
    print(f"Loaded {len(df)} completed games, seasons {sorted(df['season'].unique())}")

    train_df = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test_mask = df["season"].isin(TEST_SEASONS)
    print(f"Train: {len(train_df)} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}). "
          f"Test: {test_mask.sum()} completed {TEST_SEASONS[0]}-{TEST_SEASONS[0]+1} games")

    print("\nFitting Elo hyperparameters (K, home_adv, scale, season_regression) via grid search...")
    elo_params = fit_elo_hyperparams(train_df)
    print("Best Elo params:", elo_params)

    elo_preds_full = run_elo(df, k=elo_params["k"], home_adv=elo_params["home_adv"],
                              scale=elo_params["scale"],
                              season_regression=elo_params["season_regression"])
    elo_test_preds = elo_preds_full[test_mask.values]
    y_test = df.loc[test_mask, "home_win"].values

    result = {
        "n_games": int(len(y_test)),
        "brier": brier_score(y_test, elo_test_preds),
        "log_loss": log_loss(y_test, elo_test_preds),
        "accuracy": accuracy(y_test, elo_test_preds),
        "calibration": calibration_curve(y_test, elo_test_preds, n_bins=10),
    }

    print(f"\n=== Out-of-sample results, {TEST_SEASONS[0]}-{TEST_SEASONS[0]+1} season ===")
    print(f"  Brier score: {result['brier']:.4f}")
    print(f"  Log loss:    {result['log_loss']:.4f}")
    print(f"  Accuracy:    {result['accuracy']:.4f}")
    print("  Calibration (predicted -> actual, count):")
    for b in result["calibration"]:
        print(f"    [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}]  pred={b['predicted_mean']:.3f}  actual={b['actual_mean']:.3f}  n={b['count']}")

    # Naive baseline for comparison: home team always favored at the
    # league-wide historical home-win rate, no team skill at all -- the Elo
    # model needs to clearly beat this to be worth anything.
    naive_pred = train_df["home_win"].mean()
    naive_preds = [naive_pred] * len(y_test)
    print(f"\n  Naive baseline (flat {naive_pred:.3f} home-win rate): "
          f"Brier={brier_score(y_test, naive_preds):.4f}  LogLoss={log_loss(y_test, naive_preds):.4f}")

    out_path = ROOT / "notebooks_out" / "nhl_win_prob_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"elo_params": elo_params, "result": result}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

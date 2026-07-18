"""Backtest MLB win-probability Elo model. Train (hyperparam fitting only):
2024 season. Held-out test (all reported metrics): 2025 season (2026 is
partial/in-progress and reserved for live current-state use, not backtest)."""
import sys
import pathlib
import json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.elo_model import run_elo, fit_elo_hyperparams
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

TRAIN_SEASONS = [2023, 2024]
TEST_SEASONS = [2025]


def main():
    df = load_games()
    print(f"Loaded {len(df)} completed games, seasons {sorted(df['season'].unique())}")

    train_df = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test_mask = df["season"].isin(TEST_SEASONS)
    print(f"Train: {len(train_df)} games ({TRAIN_SEASONS}). Test: {test_mask.sum()} games ({TEST_SEASONS})")

    print("\nFitting Elo hyperparameters on train season via grid search...")
    elo_params = fit_elo_hyperparams(train_df)
    print("Best Elo params:", elo_params)

    elo_preds_full = run_elo(df, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"])
    elo_test_preds = elo_preds_full[test_mask.values]
    y_test = df.loc[test_mask, "home_win"].values

    result = {
        "n_games": int(len(y_test)),
        "brier": brier_score(y_test, elo_test_preds),
        "log_loss": log_loss(y_test, elo_test_preds),
        "accuracy": accuracy(y_test, elo_test_preds),
        "calibration": calibration_curve(y_test, elo_test_preds, n_bins=10),
    }

    print(f"\n=== Out-of-sample results, {TEST_SEASONS} test season ===")
    print(f"  Brier score: {result['brier']:.4f}")
    print(f"  Log loss:    {result['log_loss']:.4f}")
    print(f"  Accuracy:    {result['accuracy']:.4f}")
    print("  Calibration (predicted -> actual, count):")
    for b in result["calibration"]:
        print(f"    [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}]  pred={b['predicted_mean']:.3f}  actual={b['actual_mean']:.3f}  n={b['count']}")

    out_path = ROOT / "notebooks_out" / "mlb_win_prob_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"elo_params": elo_params, "result": result}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

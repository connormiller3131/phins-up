"""Does folding today's actual starting-pitcher/bullpen quality into the
projected RUN TOTAL (not just the win-probability model, which already uses
it) actually predict real runs scored better than the deployed simple
average of (own trailing runs scored, opponent trailing runs allowed)?
Tested here on real held-out 2026 data before touching production, same
"prove it helps first" approach as every other model change on this site.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.pitcher_ratings import build_sp_ratings, build_bullpen_ratings

TRAIN_SEASONS = [2024, 2025]  # SP/bullpen ratings need Statcast run_value, same constraint as the win-prob blend
TEST_SEASONS = [2026]


def build_dataset():
    games = load_games()
    sp = build_sp_ratings().drop_duplicates(["team", "game_date"], keep="first")[["team", "game_date", "sp_rating"]]
    bp = build_bullpen_ratings()

    games = games.merge(sp.rename(columns={"team": "home_team", "sp_rating": "home_sp"}), on=["home_team", "game_date"], how="left")
    games = games.merge(sp.rename(columns={"team": "away_team", "sp_rating": "away_sp"}), on=["away_team", "game_date"], how="left")
    games = games.merge(bp.rename(columns={"team": "home_team", "bullpen_rating": "home_bp"}), on=["home_team", "game_date"], how="left")
    games = games.merge(bp.rename(columns={"team": "away_team", "bullpen_rating": "away_bp"}), on=["away_team", "game_date"], how="left")

    # Long format: one row per (team, game) as the BATTING side -- the
    # opponent's SP/bullpen is what THIS team's batters actually faced.
    home_rows = games[["season", "game_date", "home_score", "home_trailing_runs_scored", "away_trailing_runs_allowed", "away_sp", "away_bp"]].rename(
        columns={"home_score": "runs", "home_trailing_runs_scored": "own_scored",
                 "away_trailing_runs_allowed": "opp_allowed", "away_sp": "opp_sp", "away_bp": "opp_bp"})
    away_rows = games[["season", "game_date", "away_score", "away_trailing_runs_scored", "home_trailing_runs_allowed", "home_sp", "home_bp"]].rename(
        columns={"away_score": "runs", "away_trailing_runs_scored": "own_scored",
                 "home_trailing_runs_allowed": "opp_allowed", "home_sp": "opp_sp", "home_bp": "opp_bp"})
    long = pd.concat([home_rows, away_rows], ignore_index=True).dropna(subset=["runs", "own_scored", "opp_allowed"])
    return long


def main():
    df = build_dataset()
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n_train={len(train)} n_test={len(test)}")

    for col in ["opp_sp", "opp_bp"]:
        fill = train[col].mean()
        fill = 0.0 if pd.isna(fill) else float(fill)
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

    baseline_pred_test = (test["own_scored"] + test["opp_allowed"]) / 2
    baseline_mae = float(np.mean(np.abs(test["runs"] - baseline_pred_test)))
    baseline_rmse = float(np.sqrt(np.mean((test["runs"] - baseline_pred_test) ** 2)))

    features = ["own_scored", "opp_allowed", "opp_sp", "opp_bp"]
    model = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model.fit(train[features].values, train["runs"].values)
    pred_test = model.predict(test[features].values)
    model_mae = float(np.mean(np.abs(test["runs"] - pred_test)))
    model_rmse = float(np.sqrt(np.mean((test["runs"] - pred_test) ** 2)))

    # Same features minus SP/bullpen -- isolates whether the ridge refit
    # ALONE (vs. the production formula's fixed 50/50 average) explains any
    # gain, separate from the SP/bullpen signal itself.
    features_no_pitcher = ["own_scored", "opp_allowed"]
    model_np = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model_np.fit(train[features_no_pitcher].values, train["runs"].values)
    pred_test_np = model_np.predict(test[features_no_pitcher].values)
    model_np_mae = float(np.mean(np.abs(test["runs"] - pred_test_np)))

    print(f"\nBaseline (production formula, own+opp_allowed)/2: MAE={baseline_mae:.4f} RMSE={baseline_rmse:.4f}")
    print(f"Ridge refit, same 2 features (no pitcher):          MAE={model_np_mae:.4f}")
    print(f"Ridge with SP/bullpen added:                         MAE={model_mae:.4f} RMSE={model_rmse:.4f}")
    print(f"\nCoefficients (with SP/bullpen): {dict(zip(features, model.coef_.tolist()))}")
    print(f"Intercept: {model.intercept_:.4f}")


if __name__ == "__main__":
    main()

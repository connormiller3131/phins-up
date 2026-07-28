"""Does the opposing starting goalie's real recent quality (goals saved
above average, derived from real per-game boxscore data pulled just for
this) predict a team's real goals scored better than the deployed simple
average of (own trailing goals scored, opponent trailing goals allowed)?
Same "prove it first" approach as MLB's backtest_runs_model.py and NFL's
backtest_points_model.py.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nhl.games import load_games
from pipeline.nhl.goalie_ratings import build_goalie_ratings

TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
TEST_SEASONS = [2025]


def build_dataset():
    games = load_games()
    games["game_date"] = pd.to_datetime(games["game_date"])
    ratings, _ = build_goalie_ratings()
    ratings["game_date"] = pd.to_datetime(ratings["game_date"])
    home_r = ratings.rename(columns={"team": "home_team", "sg_rating": "home_sg"})[["home_team", "game_date", "home_sg"]]
    away_r = ratings.rename(columns={"team": "away_team", "sg_rating": "away_sg"})[["away_team", "game_date", "away_sg"]]
    games = games.merge(home_r, on=["home_team", "game_date"], how="left")
    games = games.merge(away_r, on=["away_team", "game_date"], how="left")

    home_rows = games[["season", "home_score", "home_trailing_goals_scored", "away_trailing_goals_allowed", "away_sg"]].rename(
        columns={"home_score": "goals", "home_trailing_goals_scored": "own_scored",
                 "away_trailing_goals_allowed": "opp_allowed", "away_sg": "opp_goalie"})
    away_rows = games[["season", "away_score", "away_trailing_goals_scored", "home_trailing_goals_allowed", "home_sg"]].rename(
        columns={"away_score": "goals", "away_trailing_goals_scored": "own_scored",
                 "home_trailing_goals_allowed": "opp_allowed", "home_sg": "opp_goalie"})
    long = pd.concat([home_rows, away_rows], ignore_index=True)
    return long.dropna(subset=["goals", "own_scored", "opp_allowed"])


def main():
    df = build_dataset()
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n_train={len(train)} n_test={len(test)}")
    print(f"opp_goalie coverage in test: {test['opp_goalie'].notna().mean()*100:.1f}%")

    fill = train["opp_goalie"].mean()
    fill = 0.0 if pd.isna(fill) else float(fill)
    train["opp_goalie"] = train["opp_goalie"].fillna(fill)
    test["opp_goalie"] = test["opp_goalie"].fillna(fill)

    baseline_pred_test = (test["own_scored"] + test["opp_allowed"]) / 2
    baseline_mae = float(np.mean(np.abs(test["goals"] - baseline_pred_test)))

    features_np = ["own_scored", "opp_allowed"]
    model_np = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model_np.fit(train[features_np].values, train["goals"].values)
    pred_np = model_np.predict(test[features_np].values)
    model_np_mae = float(np.mean(np.abs(test["goals"] - pred_np)))

    features = ["own_scored", "opp_allowed", "opp_goalie"]
    model = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model.fit(train[features].values, train["goals"].values)
    pred_test = model.predict(test[features].values)
    model_mae = float(np.mean(np.abs(test["goals"] - pred_test)))

    print(f"\nBaseline (production formula, own+opp_allowed)/2: MAE={baseline_mae:.4f}")
    print(f"Ridge refit, same 2 features (no goalie):          MAE={model_np_mae:.4f}")
    print(f"Ridge with opposing starting goalie added:         MAE={model_mae:.4f}")
    print(f"\nCoefficients: {dict(zip(features, model.coef_.tolist()))}")


if __name__ == "__main__":
    main()

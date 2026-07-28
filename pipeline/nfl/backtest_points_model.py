"""Does folding real recent offense/defense efficiency (pass yards/attempt,
rush yards/carry -- the same signal team_offense_defense.py already
computed and validated for the Elo win-probability blend, though that blend
was never actually wired into the deployed model) into the projected POINT
total predict real points scored better than the deployed simple average of
(own trailing points scored, opponent trailing points allowed)? Same
"prove it first" approach as MLB's backtest_runs_model.py.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.games import load_games
from pipeline.nfl.team_offense_defense import build_offense_defense_ratings

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_SEASONS = [2025]
SCORING_WINDOW, SCORING_MIN_GAMES = 8, 3


def build_scoring_trailing(games_df):
    home = games_df[["season", "week", "game_id", "game_date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "allowed"})
    away = games_df[["season", "week", "game_id", "game_date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "allowed"})
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "game_date"])
    long["trailing_scored"] = long.groupby("team")["scored"].transform(
        lambda s: s.shift(1).rolling(SCORING_WINDOW, min_periods=SCORING_MIN_GAMES).mean())
    long["trailing_allowed"] = long.groupby("team")["allowed"].transform(
        lambda s: s.shift(1).rolling(SCORING_WINDOW, min_periods=SCORING_MIN_GAMES).mean())
    return long


def build_dataset():
    games_df = load_games()
    scoring = build_scoring_trailing(games_df)

    ratings = build_offense_defense_ratings()
    opp = ratings[["game_id", "team", "pass_ypa_def_trail", "rush_ypc_def_trail"]].rename(
        columns={"team": "_other_team", "pass_ypa_def_trail": "opp_pass_ypa_def", "rush_ypc_def_trail": "opp_rush_ypc_def"})
    merged = ratings.merge(opp, on="game_id")
    merged = merged[merged["team"] != merged["_other_team"]].drop(columns=["_other_team"])

    df = scoring.merge(
        merged[["team", "season", "week", "game_id", "pass_ypa_off_trail", "rush_ypc_off_trail", "opp_pass_ypa_def", "opp_rush_ypc_def"]],
        on=["team", "season", "week", "game_id"], how="left",
    )
    df = df.rename(columns={"pass_ypa_off_trail": "own_pass_ypa_off", "rush_ypc_off_trail": "own_rush_ypc_off"})
    return df.dropna(subset=["scored", "trailing_scored", "trailing_allowed"])


def main():
    df = build_dataset()
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n_train={len(train)} n_test={len(test)}")

    for col in ["own_pass_ypa_off", "own_rush_ypc_off", "opp_pass_ypa_def", "opp_rush_ypc_def"]:
        fill = train[col].mean()
        fill = 0.0 if pd.isna(fill) else float(fill)
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

    baseline_pred_test = (test["trailing_scored"] + test["trailing_allowed"]) / 2
    baseline_mae = float(np.mean(np.abs(test["scored"] - baseline_pred_test)))

    features = ["trailing_scored", "trailing_allowed", "own_pass_ypa_off", "own_rush_ypc_off", "opp_pass_ypa_def", "opp_rush_ypc_def"]
    model = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model.fit(train[features].values, train["scored"].values)
    pred_test = model.predict(test[features].values)
    model_mae = float(np.mean(np.abs(test["scored"] - pred_test)))

    features_np = ["trailing_scored", "trailing_allowed"]
    model_np = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model_np.fit(train[features_np].values, train["scored"].values)
    pred_test_np = model_np.predict(test[features_np].values)
    model_np_mae = float(np.mean(np.abs(test["scored"] - pred_test_np)))

    print(f"\nBaseline (production formula, own+opp_allowed)/2:  MAE={baseline_mae:.4f}")
    print(f"Ridge refit, same 2 features (no efficiency):       MAE={model_np_mae:.4f}")
    print(f"Ridge with offense/defense efficiency added:        MAE={model_mae:.4f}")
    print(f"\nCoefficients: {dict(zip(features, model.coef_.tolist()))}")


if __name__ == "__main__":
    main()

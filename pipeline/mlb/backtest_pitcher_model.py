"""Backtest whether adding starting-pitcher and bullpen strength to the
win-probability model actually improves it, on the same 2019-2025 train /
2026 held-out test split as the team-only Elo backtest.

Approach: treat the existing (already-validated) Elo win probability as one
input to a small logistic-regression blend, alongside SP-rating-diff and
bullpen-rating-diff -- rather than re-deriving team strength from scratch,
this stacks the new pitcher/bullpen signal on top of what Elo already gets
right. Missing ratings (rookies, insufficient trailing starts) are imputed
with the training-set mean, not dropped, so every game gets a prediction.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.elo_model import run_elo
from pipeline.mlb.pitcher_ratings import build_sp_ratings, build_bullpen_ratings
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

# The base Elo (elo_pred, used as one input feature below) was fit on the
# full 2019-2025 team-schedule history. But pitcher-level Statcast data only
# covers 2024-2026 -- the SP/bullpen blend's own training window is
# necessarily narrower than Elo's. Extending pitcher Statcast back to
# 2019-2023 is a real future improvement (another ~25-35min pull) but not
# done here.
TRAIN_SEASONS = [2024, 2025]
TEST_SEASONS = [2026]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    with open(ROOT / "notebooks_out" / "mlb_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    games = load_games()
    print(f"Loaded {len(games)} games, seasons {sorted(games['season'].unique())}")

    elo_pred = run_elo(games, k=elo_params["k"], home_adv=elo_params["home_adv"],
                       scale=elo_params["scale"], season_regression=elo_params["season_regression"])
    games = games.assign(elo_pred=elo_pred)

    sp_ratings = build_sp_ratings()  # walk-forward: rating AS OF entering that start
    bp_ratings = build_bullpen_ratings()

    print("Attaching starting pitcher / bullpen ratings to each game (vectorized merge)...")

    # sp_ratings already has one row per (player_id, team, game_date) for
    # starts; a team can only have started one pitcher per date except
    # doubleheaders, which we don't disambiguate -- keep the first.
    sp_by_team_date = sp_ratings.drop_duplicates(["team", "game_date"], keep="first")[
        ["team", "game_date", "sp_rating"]]

    games = games.merge(sp_by_team_date.rename(columns={"team": "home_team", "sp_rating": "home_sp"}),
                        on=["home_team", "game_date"], how="left")
    games = games.merge(sp_by_team_date.rename(columns={"team": "away_team", "sp_rating": "away_sp"}),
                        on=["away_team", "game_date"], how="left")
    games = games.merge(bp_ratings.rename(columns={"team": "home_team", "bullpen_rating": "home_bp"}),
                        on=["home_team", "game_date"], how="left")
    games = games.merge(bp_ratings.rename(columns={"team": "away_team", "bullpen_rating": "away_bp"}),
                        on=["away_team", "game_date"], how="left")
    games["sp_diff"] = games["home_sp"] - games["away_sp"]
    games["bp_diff"] = games["home_bp"] - games["away_bp"]

    train = games[games["season"].isin(TRAIN_SEASONS)].copy()
    test = games[games["season"].isin(TEST_SEASONS)].copy()
    print(f"Train: {len(train)} games. Test: {len(test)} games.")
    print(f"  Train SP coverage: {train['sp_diff'].notna().mean():.1%}, BP coverage: {train['bp_diff'].notna().mean():.1%}")

    sp_fill = train["sp_diff"].mean()
    bp_fill = train["bp_diff"].mean()
    for df in (train, test):
        df["sp_diff"] = df["sp_diff"].fillna(sp_fill)
        df["bp_diff"] = df["bp_diff"].fillna(bp_fill)
        df["elo_logit"] = logit(df["elo_pred"].values)

    X_train = train[["elo_logit", "sp_diff", "bp_diff"]].values
    y_train = train["home_win"].values
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(X_train, y_train)
    print("Fitted blend coefficients [elo_logit, sp_diff, bp_diff]:", model.coef_[0], "intercept:", model.intercept_[0])

    X_test = test[["elo_logit", "sp_diff", "bp_diff"]].values
    y_test = test["home_win"].values
    blend_pred = model.predict_proba(X_test)[:, 1]
    elo_only_pred = test["elo_pred"].values

    for name, preds in [("elo_only (baseline)", elo_only_pred), ("elo+SP+bullpen (new)", blend_pred)]:
        print(f"\n{name}")
        print(f"  Brier:    {brier_score(y_test, preds):.4f}")
        print(f"  Log loss: {log_loss(y_test, preds):.4f}")
        print(f"  Accuracy: {accuracy(y_test, preds):.4f}")

    out_path = ROOT / "notebooks_out" / "mlb_pitcher_model_backtest.json"
    with open(out_path, "w") as f:
        json.dump({
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "C": float(model.C_[0]), "sp_fill": float(sp_fill), "bp_fill": float(bp_fill),
            "train_sp_coverage": float(train["sp_diff"].notna().mean()),
            "elo_only": {"brier": brier_score(y_test, elo_only_pred), "log_loss": log_loss(y_test, elo_only_pred),
                        "accuracy": accuracy(y_test, elo_only_pred)},
            "blend": {"brier": brier_score(y_test, blend_pred), "log_loss": log_loss(y_test, blend_pred),
                     "accuracy": accuracy(y_test, blend_pred)},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

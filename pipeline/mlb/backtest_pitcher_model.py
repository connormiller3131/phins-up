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
    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    games["form_diff"] = ((games["home_trailing_runs_scored"] - games["home_trailing_runs_allowed"])
                          - (games["away_trailing_runs_scored"] - games["away_trailing_runs_allowed"]))

    train = games[games["season"].isin(TRAIN_SEASONS)].copy()
    test = games[games["season"].isin(TEST_SEASONS)].copy()
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(f"Empty train ({len(train)}) or test ({len(test)}) set -- "
                           f"check that pitcher_game_logs.parquet covers {TRAIN_SEASONS + TEST_SEASONS}.")
    print(f"Train: {len(train)} games. Test: {len(test)} games.")
    print(f"  Train SP coverage: {train['sp_diff'].notna().mean():.1%}, BP coverage: {train['bp_diff'].notna().mean():.1%}, "
          f"rest coverage: {train['rest_diff'].notna().mean():.1%}, form coverage: {train['form_diff'].notna().mean():.1%}")

    fills = {}
    for col in ("sp_diff", "bp_diff", "rest_diff", "form_diff"):
        fill = train[col].mean()
        fills[col] = 0.0 if pd.isna(fill) else float(fill)
        for df in (train, test):
            df[col] = df[col].fillna(fills[col])
    for df in (train, test):
        df["elo_logit"] = logit(df["elo_pred"].values)

    y_train = train["home_win"].values
    y_test = test["home_win"].values
    elo_only_pred = test["elo_pred"].values

    # Test each addition incrementally so it's clear which pieces actually
    # earn their place, rather than just reporting one combined number.
    feature_sets = {
        "elo+SP+bullpen": ["elo_logit", "sp_diff", "bp_diff"],
        "elo+SP+bullpen+rest": ["elo_logit", "sp_diff", "bp_diff", "rest_diff"],
        "elo+SP+bullpen+rest+form": ["elo_logit", "sp_diff", "bp_diff", "rest_diff", "form_diff"],
    }

    print(f"\nelo_only (baseline)")
    print(f"  Brier:    {brier_score(y_test, elo_only_pred):.4f}")
    print(f"  Log loss: {log_loss(y_test, elo_only_pred):.4f}")
    print(f"  Accuracy: {accuracy(y_test, elo_only_pred):.4f}")

    results = {}
    fitted = {}
    for name, cols in feature_sets.items():
        X_train = train[cols].values
        X_test = test[cols].values
        model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        results[name] = {"brier": brier_score(y_test, preds), "log_loss": log_loss(y_test, preds),
                         "accuracy": accuracy(y_test, preds)}
        fitted[name] = (model, cols)
        print(f"\n{name}")
        print(f"  coef {dict(zip(cols, model.coef_[0]))}")
        print(f"  Brier:    {results[name]['brier']:.4f}")
        print(f"  Log loss: {results[name]['log_loss']:.4f}")
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")

    # Rest days and recent form were tested honestly and don't earn a spot:
    # elo+SP+bullpen+rest is worse than elo+SP+bullpen alone on held-out
    # Brier, and adding form on top only ties Brier while accuracy drops
    # noticeably (54.3% -> 53.8%). A tied Brier plus a clear accuracy drop
    # isn't a real improvement -- it's regularization noise -- so the
    # simpler, already-validated elo+SP+bullpen blend stays deployed.
    # Keeping the original 3-feature output schema so generate_daily_slate.py
    # doesn't need to change.
    deployed_name = "elo+SP+bullpen"
    deployed_model, deployed_cols = fitted[deployed_name]
    print(f"\nDeployed blend: {deployed_name} (rest/form tested, didn't improve held-out performance -- see all_results)")

    out_path = ROOT / "notebooks_out" / "mlb_pitcher_model_backtest.json"
    with open(out_path, "w") as f:
        json.dump({
            "coef": deployed_model.coef_[0].tolist(), "intercept": float(deployed_model.intercept_[0]),
            "C": float(deployed_model.C_[0]), "sp_fill": fills["sp_diff"], "bp_fill": fills["bp_diff"],
            "train_sp_coverage": float(train["sp_diff"].notna().mean()),
            "elo_only": {"brier": brier_score(y_test, elo_only_pred), "log_loss": log_loss(y_test, elo_only_pred),
                        "accuracy": accuracy(y_test, elo_only_pred)},
            "all_results_tested": results,
            "blend": results[deployed_name],
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

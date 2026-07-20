"""Test whether real passing/rushing offense and defense quality (yards per
attempt/carry, both for and against) and a net interception margin actually
improve on team-only Elo for NFL win probability -- same incremental,
honest-reporting methodology as MLB's backtest_pitcher_model.py: each
candidate feature set is fit and evaluated out-of-sample, and only kept if it
actually beats the Elo-only baseline. Nothing here is auto-deployed by
running this script; see the printed comparison and DEPLOYED_FEATURES for
what actually shipped.

Train (hyperparam fitting only): 2019-2024
Held-out test (all reported metrics): 2025
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.games import load_games
from pipeline.nfl.elo_model import run_elo
from pipeline.nfl.team_offense_defense import build_offense_defense_ratings
from pipeline.common.metrics import brier_score, log_loss, accuracy

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_SEASONS = [2025]

# Empty until a feature set actually beats the elo_only baseline below --
# deliberately not auto-selected from whichever result happens to score best,
# same discipline as MLB's blend.
DEPLOYED_FEATURES = []


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    games = load_games()
    print(f"Loaded {len(games)} games, seasons {sorted(games['season'].unique())}")

    elo_pred = run_elo(games, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"],
                       rest_adv=elo_params.get("rest_adv", 0.0), season_regression=elo_params.get("season_regression", 0.75))
    games = games.assign(elo_pred=elo_pred)

    ratings = build_offense_defense_ratings()
    print("Attaching trailing offense/defense ratings to each game...")

    rating_cols = ["pass_ypa_off_trail", "rush_ypc_off_trail", "pass_ypa_def_trail", "rush_ypc_def_trail",
                   "pass_epa_pp_off_trail", "rush_epa_pp_off_trail", "pass_epa_pp_def_trail", "rush_epa_pp_def_trail",
                   "int_margin_trail"]
    home_r = ratings.rename(columns={"team": "home_team", **{c: "home_" + c.replace("_trail", "") for c in rating_cols}})
    away_r = ratings.rename(columns={"team": "away_team", **{c: "away_" + c.replace("_trail", "") for c in rating_cols}})
    home_cols = ["home_" + c.replace("_trail", "") for c in rating_cols]
    away_cols = ["away_" + c.replace("_trail", "") for c in rating_cols]

    games = games.merge(home_r[["game_id", "home_team"] + home_cols], on=["game_id", "home_team"], how="left")
    games = games.merge(away_r[["game_id", "away_team"] + away_cols], on=["game_id", "away_team"], how="left")

    # Signed so positive always favors the home team: for offense, higher
    # own rate is better; for defense (an ALLOWED rate), lower is better, so
    # the away team's allowed rate minus the home team's is what's positive
    # when home's defense is stingier.
    games["pass_off_diff"] = games["home_pass_ypa_off"] - games["away_pass_ypa_off"]
    games["rush_off_diff"] = games["home_rush_ypc_off"] - games["away_rush_ypc_off"]
    games["pass_def_diff"] = games["away_pass_ypa_def"] - games["home_pass_ypa_def"]
    games["rush_def_diff"] = games["away_rush_ypc_def"] - games["home_rush_ypc_def"]
    games["pass_epa_off_diff"] = games["home_pass_epa_pp_off"] - games["away_pass_epa_pp_off"]
    games["rush_epa_off_diff"] = games["home_rush_epa_pp_off"] - games["away_rush_epa_pp_off"]
    games["pass_epa_def_diff"] = games["away_pass_epa_pp_def"] - games["home_pass_epa_pp_def"]
    games["rush_epa_def_diff"] = games["away_rush_epa_pp_def"] - games["home_rush_epa_pp_def"]
    games["int_margin_diff"] = games["home_int_margin"] - games["away_int_margin"]

    train = games[games["season"].isin(TRAIN_SEASONS)].copy()
    test = games[games["season"].isin(TEST_SEASONS)].copy()
    print(f"Train: {len(train)} games. Test: {len(test)} games.")

    feat_cols = ["pass_off_diff", "rush_off_diff", "pass_def_diff", "rush_def_diff",
                 "pass_epa_off_diff", "rush_epa_off_diff", "pass_epa_def_diff", "rush_epa_def_diff", "int_margin_diff"]
    print(f"  Train coverage: " + ", ".join(f"{c}={train[c].notna().mean():.1%}" for c in feat_cols))

    fills = {}
    for col in feat_cols:
        fill = train[col].mean()
        fills[col] = 0.0 if pd.isna(fill) else float(fill)
        for df in (train, test):
            df[col] = df[col].fillna(fills[col])
    for df in (train, test):
        df["elo_logit"] = logit(df["elo_pred"].values)

    # Ties (home_win == 0.5) aren't a valid binary-classifier label -- excluded
    # from fitting, not just from the test-set loss calc (the earlier crash
    # here: LogisticRegressionCV rejected a continuous-valued y_train).
    train = train[train["home_win"] != 0.5].copy()
    y_train = train["home_win"].values
    non_tie_test = test["home_win"] != 0.5
    y_test = test.loc[non_tie_test, "home_win"].values
    elo_only_pred = test.loc[non_tie_test, "elo_pred"].values

    feature_sets = {
        "elo+offense yards (pass+rush for)": ["elo_logit", "pass_off_diff", "rush_off_diff"],
        "elo+defense yards (pass+rush allowed)": ["elo_logit", "pass_def_diff", "rush_def_diff"],
        "elo+int_margin": ["elo_logit", "int_margin_diff"],
        "elo+everything yards": ["elo_logit", "pass_off_diff", "rush_off_diff", "pass_def_diff", "rush_def_diff", "int_margin_diff"],
        "elo+offense EPA (pass+rush for)": ["elo_logit", "pass_epa_off_diff", "rush_epa_off_diff"],
        "elo+defense EPA (pass+rush allowed)": ["elo_logit", "pass_epa_def_diff", "rush_epa_def_diff"],
        "elo+everything EPA": ["elo_logit", "pass_epa_off_diff", "rush_epa_off_diff", "pass_epa_def_diff", "rush_epa_def_diff", "int_margin_diff"],
    }

    print(f"\nelo_only (baseline)")
    print(f"  Brier:    {brier_score(y_test, elo_only_pred):.4f}")
    print(f"  Log loss: {log_loss(y_test, elo_only_pred):.4f}")
    print(f"  Accuracy: {accuracy(y_test, elo_only_pred):.4f}")

    results = {}
    fitted = {}
    for name, cols in feature_sets.items():
        X_train = train[cols].values
        X_test = test.loc[non_tie_test, cols].values
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

    elo_only_results = {"brier": brier_score(y_test, elo_only_pred), "log_loss": log_loss(y_test, elo_only_pred),
                        "accuracy": accuracy(y_test, elo_only_pred)}

    out_path = ROOT / "notebooks_out" / "nfl_offense_defense_backtest.json"
    with open(out_path, "w") as f:
        json.dump({
            "fills": fills,
            "elo_only": elo_only_results,
            "all_results_tested": results,
            "deployed_features": DEPLOYED_FEATURES,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"\nDEPLOYED_FEATURES = {DEPLOYED_FEATURES} -- update this constant and rerun once you've decided what (if anything) to deploy.")


if __name__ == "__main__":
    main()

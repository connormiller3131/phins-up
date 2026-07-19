"""Experimental feature engineering for the MLB win-probability model, on
top of the already-fitted Elo baseline. Each candidate is tested
incrementally in a small logistic blend (same architecture as the SP/
bullpen blend) and reported honestly -- only kept if it actually improves
held-out performance. Nothing here changes what's deployed; that only
happens if backtest_pitcher_model.py (or a successor) is updated to include
a proven feature.

Candidates tested:
1. entering_streak_diff -- each team's win/loss streak walking INTO the
   game (Baseball-Reference's own Streak column reflects the streak AFTER
   that game, i.e. it encodes that game's own result -- shifted by one
   game per team here, verified against a real sequence before trusting
   it, or this would have been a leakage bug).
2. division_game -- whether the two teams are in the same division (real,
   fixed 2025-current alignment, not inferred).
3. team offensive quality via trailing PA-weighted est_woba (Statcast's
   own expected-outcome-from-contact-quality metric), a less noisy
   alternative to the raw-runs "recent form" feature that already failed.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.elo_model import run_elo
from pipeline.mlb.team_map import br_to_statcast
from pipeline.mlb.pitcher_ratings import build_sp_ratings, build_bullpen_ratings
from pipeline.common.metrics import brier_score, log_loss, accuracy

DATA_DIR = ROOT / "data" / "mlb"
TRAIN_SEASONS_FULL = [2019, 2020, 2021, 2022, 2023, 2024, 2025]  # for streak/division (no Statcast needed)
TRAIN_SEASONS_SP = [2024, 2025]  # for offense-quality (constrained by Statcast coverage)
TEST_SEASONS = [2026]
FORM_WINDOW, FORM_MIN_GAMES = 10, 3

DIVISIONS = {
    "NYY": "AL East", "BOS": "AL East", "TOR": "AL East", "BAL": "AL East", "TB": "AL East",
    "CLE": "AL Central", "MIN": "AL Central", "KC": "AL Central", "CWS": "AL Central", "DET": "AL Central",
    "HOU": "AL West", "SEA": "AL West", "TEX": "AL West", "LAA": "AL West", "ATH": "AL West",
    "ATL": "NL East", "PHI": "NL East", "NYM": "NL East", "MIA": "NL East", "WSH": "NL East",
    "MIL": "NL Central", "CHC": "NL Central", "STL": "NL Central", "CIN": "NL Central", "PIT": "NL Central",
    "LAD": "NL West", "SD": "NL West", "AZ": "NL West", "SF": "NL West", "COL": "NL West",
}


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def add_entering_streak(games):
    raw = pd.read_parquet(DATA_DIR / "team_schedule_raw.parquet")
    raw["team"] = raw["team"].map(br_to_statcast)
    from pipeline.mlb.games import _parse_date
    raw["game_date"] = raw.apply(lambda r: _parse_date(r["Date"], r["season"]), axis=1)
    raw = raw.dropna(subset=["game_date", "Streak"]).sort_values(["team", "game_date"])
    raw["_occ"] = raw.groupby(["team", "game_date"]).cumcount()
    raw["entering_streak"] = raw.groupby("team")["Streak"].shift(1)
    streak = raw[["team", "game_date", "_occ", "entering_streak"]].drop_duplicates(["team", "game_date", "_occ"])

    games = games.copy()
    games["_home_occ"] = games.groupby(["home_team", "game_date"]).cumcount()
    games["_away_occ"] = games.groupby(["away_team", "game_date"]).cumcount()
    games = games.merge(streak.rename(columns={"team": "home_team", "_occ": "_home_occ", "entering_streak": "home_streak"}),
                        on=["home_team", "game_date", "_home_occ"], how="left")
    games = games.merge(streak.rename(columns={"team": "away_team", "_occ": "_away_occ", "entering_streak": "away_streak"}),
                        on=["away_team", "game_date", "_away_occ"], how="left")
    games["streak_diff"] = games["home_streak"] - games["away_streak"]
    return games.drop(columns=["_home_occ", "_away_occ"])


def add_division_flag(games):
    games = games.copy()
    games["division_game"] = (
        games["home_team"].map(DIVISIONS) == games["away_team"].map(DIVISIONS)
    ).astype(float)
    return games


def add_offense_quality(games):
    """PA-weighted team est_woba per game, trailing average, walk-forward
    safe -- a Statcast-quality-of-contact signal instead of raw runs
    (which already failed as a 'recent form' feature)."""
    bg = pl.read_parquet(DATA_DIR / "batter_game_logs.parquet").to_pandas()
    bg = bg.dropna(subset=["est_woba", "pa_count"])
    bg["woba_x_pa"] = bg["est_woba"] * bg["pa_count"]
    team_game = (
        bg.groupby(["team", "game_date"])
        .agg(woba_x_pa=("woba_x_pa", "sum"), pa=("pa_count", "sum"))
        .reset_index()
    )
    team_game["team_woba"] = team_game["woba_x_pa"] / team_game["pa"]
    team_game = team_game.sort_values(["team", "game_date"])
    team_game["_occ"] = team_game.groupby(["team", "game_date"]).cumcount()
    team_game["trailing_woba"] = team_game.groupby("team")["team_woba"].transform(
        lambda s: s.shift(1).rolling(window=FORM_WINDOW, min_periods=FORM_MIN_GAMES).mean()
    )
    twoba = team_game[["team", "game_date", "_occ", "trailing_woba"]]

    games = games.copy()
    games["_home_occ"] = games.groupby(["home_team", "game_date"]).cumcount()
    games["_away_occ"] = games.groupby(["away_team", "game_date"]).cumcount()
    games = games.merge(twoba.rename(columns={"team": "home_team", "_occ": "_home_occ", "trailing_woba": "home_woba"}),
                        on=["home_team", "game_date", "_home_occ"], how="left")
    games = games.merge(twoba.rename(columns={"team": "away_team", "_occ": "_away_occ", "trailing_woba": "away_woba"}),
                        on=["away_team", "game_date", "_away_occ"], how="left")
    games["woba_diff"] = games["home_woba"] - games["away_woba"]
    return games.drop(columns=["_home_occ", "_away_occ"])


def add_bullpen_workload(games):
    """Real fatigue signal distinct from the existing quality rating: total
    outs recorded by a team's relievers in the trailing 2 calendar days
    (using the outs_recorded column, itself real innings pitched in
    disguise -- verified against a real box score when it was added). A
    bullpen that just threw heavy innings is more taxed than a fresh one,
    regardless of how GOOD those relievers have looked lately."""
    pg = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    bp = pg[~pg["is_starter"]].copy()
    team_day = bp.groupby(["team", "game_date"])["outs_recorded"].sum().reset_index()
    team_day = team_day.sort_values(["team", "game_date"]).set_index("game_date")

    workload = (
        team_day.groupby("team")["outs_recorded"]
        .rolling("2D", closed="left").sum()  # strictly before the current date -- no leakage
        .reset_index()
        .rename(columns={"outs_recorded": "bp_workload_2d"})
    )

    games = games.copy()
    games = games.merge(workload.rename(columns={"team": "home_team", "bp_workload_2d": "home_bp_workload"}),
                        on=["home_team", "game_date"], how="left")
    games = games.merge(workload.rename(columns={"team": "away_team", "bp_workload_2d": "away_bp_workload"}),
                        on=["away_team", "game_date"], how="left")
    games["bp_workload_diff"] = games["home_bp_workload"] - games["away_bp_workload"]
    return games


def fit_and_eval(train, test, cols, y_train_col="home_win"):
    fills = {}
    for col in cols:
        if col == "elo_logit":
            continue
        fill = train[col].mean()
        fills[col] = 0.0 if pd.isna(fill) else float(fill)
        train[col] = train[col].fillna(fills[col])
        test[col] = test[col].fillna(fills[col])
    X_train, y_train = train[cols].values, train[y_train_col].values
    X_test, y_test = test[cols].values, test[y_train_col].values
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return {
        "coef": dict(zip(cols, model.coef_[0].tolist())),
        "brier": brier_score(y_test, preds), "log_loss": log_loss(y_test, preds), "accuracy": accuracy(y_test, preds),
    }


def main():
    with open(ROOT / "notebooks_out" / "mlb_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    games = load_games()
    elo_pred = run_elo(games, k=elo_params["k"], home_adv=elo_params["home_adv"],
                       scale=elo_params["scale"], season_regression=elo_params["season_regression"])
    games = games.assign(elo_pred=elo_pred, elo_logit=logit(elo_pred))

    games = add_entering_streak(games)
    games = add_division_flag(games)

    results = {}

    # --- Track 1: streak + division, big sample (2019-2025 train), no Statcast needed ---
    t1 = games[games["season"].isin(TRAIN_SEASONS_FULL)].copy()
    test1 = games[games["season"].isin(TEST_SEASONS)].copy()
    print(f"Track 1 (streak/division): train={len(t1)}, test={len(test1)}")

    base = fit_and_eval(t1.copy(), test1.copy(), ["elo_logit"])
    results["elo_only"] = base
    print(f"\nelo_only: Brier={base['brier']:.4f} LogLoss={base['log_loss']:.4f} Acc={base['accuracy']:.4f}")

    r = fit_and_eval(t1.copy(), test1.copy(), ["elo_logit", "streak_diff"])
    results["elo+streak"] = r
    print(f"elo+streak: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    r = fit_and_eval(t1.copy(), test1.copy(), ["elo_logit", "division_game"])
    results["elo+division"] = r
    print(f"elo+division: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    r = fit_and_eval(t1.copy(), test1.copy(), ["elo_logit", "streak_diff", "division_game"])
    results["elo+streak+division"] = r
    print(f"elo+streak+division: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    # --- Track 2: offense quality (est_woba), on top of the deployed SP/bullpen blend ---
    games = add_offense_quality(games)
    sp_ratings = build_sp_ratings()
    bp_ratings = build_bullpen_ratings()
    sp_by_team_date = sp_ratings.drop_duplicates(["team", "game_date"], keep="first")[["team", "game_date", "sp_rating"]]
    games = games.merge(sp_by_team_date.rename(columns={"team": "home_team", "sp_rating": "home_sp"}), on=["home_team", "game_date"], how="left")
    games = games.merge(sp_by_team_date.rename(columns={"team": "away_team", "sp_rating": "away_sp"}), on=["away_team", "game_date"], how="left")
    games = games.merge(bp_ratings.rename(columns={"team": "home_team", "bullpen_rating": "home_bp"}), on=["home_team", "game_date"], how="left")
    games = games.merge(bp_ratings.rename(columns={"team": "away_team", "bullpen_rating": "away_bp"}), on=["away_team", "game_date"], how="left")
    games["sp_diff"] = games["home_sp"] - games["away_sp"]
    games["bp_diff"] = games["home_bp"] - games["away_bp"]

    t2 = games[games["season"].isin(TRAIN_SEASONS_SP)].copy()
    test2 = games[games["season"].isin(TEST_SEASONS)].copy()
    print(f"\nTrack 2 (offense quality): train={len(t2)}, test={len(test2)}")

    r = fit_and_eval(t2.copy(), test2.copy(), ["elo_logit", "sp_diff", "bp_diff"])
    results["elo+SP+bullpen (deployed baseline)"] = r
    print(f"\nelo+SP+bullpen (deployed baseline): Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}")

    r = fit_and_eval(t2.copy(), test2.copy(), ["elo_logit", "sp_diff", "bp_diff", "woba_diff"])
    results["elo+SP+bullpen+offense_quality"] = r
    print(f"elo+SP+bullpen+offense_quality: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    r = fit_and_eval(t2.copy(), test2.copy(), ["elo_logit", "sp_diff", "bp_diff", "woba_diff", "division_game"])
    results["elo+SP+bullpen+offense_quality+division"] = r
    print(f"elo+SP+bullpen+offense_quality+division: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    # --- Bullpen recent workload (fatigue), distinct from the existing quality rating ---
    games = add_bullpen_workload(games)
    t3 = games[games["season"].isin(TRAIN_SEASONS_SP)].copy()
    test3 = games[games["season"].isin(TEST_SEASONS)].copy()

    r = fit_and_eval(t3.copy(), test3.copy(), ["elo_logit", "sp_diff", "bp_diff", "bp_workload_diff"])
    results["elo+SP+bullpen+bp_workload"] = r
    print(f"elo+SP+bullpen+bp_workload: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    r = fit_and_eval(t3.copy(), test3.copy(), ["elo_logit", "sp_diff", "bp_diff", "woba_diff", "division_game", "bp_workload_diff"])
    results["elo+everything"] = r
    print(f"elo+everything: Brier={r['brier']:.4f} LogLoss={r['log_loss']:.4f} Acc={r['accuracy']:.4f}  coef={r['coef']}")

    # --- Does a nonlinear model resolve the Brier-vs-accuracy tension on the
    # best linear feature set (elo+SP+bullpen+offense_quality+division), or
    # is more model flexibility just overfitting a ~4800-game sample? ---
    best_cols = ["elo_logit", "sp_diff", "bp_diff", "woba_diff", "division_game"]
    tb = t2.copy()
    for col in best_cols:
        if col != "elo_logit":
            fill = tb[col].mean()
            tb[col] = tb[col].fillna(0.0 if pd.isna(fill) else fill)
    teb = test2.copy()
    for col in best_cols:
        if col != "elo_logit":
            fill = tb[col].mean()
            teb[col] = teb[col].fillna(0.0 if pd.isna(fill) else fill)

    gbm_grid = GridSearchCV(
        HistGradientBoostingClassifier(random_state=0),
        param_grid={"max_leaf_nodes": (3, 7, 15), "learning_rate": (0.01, 0.05, 0.1), "min_samples_leaf": (50, 100, 200)},
        scoring="neg_log_loss", cv=5,
    )
    gbm_grid.fit(tb[best_cols].values, tb["home_win"].values)
    gbm_preds = gbm_grid.predict_proba(teb[best_cols].values)[:, 1]
    y_test2 = teb["home_win"].values
    results["elo+SP+bullpen+offense_quality+division (gradient boosted)"] = {
        "best_params": gbm_grid.best_params_,
        "brier": brier_score(y_test2, gbm_preds), "log_loss": log_loss(y_test2, gbm_preds), "accuracy": accuracy(y_test2, gbm_preds),
    }
    print(f"\ngradient-boosted (same features): Brier={brier_score(y_test2, gbm_preds):.4f} "
          f"LogLoss={log_loss(y_test2, gbm_preds):.4f} Acc={accuracy(y_test2, gbm_preds):.4f}  best_params={gbm_grid.best_params_}")

    out_path = ROOT / "notebooks_out" / "mlb_feature_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

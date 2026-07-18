"""Backtest the Poisson run-scoring simulation's win probability against
actual 2026 outcomes, and against the existing Elo and Elo+pitcher models,
on the identical holdout. Two versions tested: pure offense/defense, and
offense/defense + starting-pitcher/bullpen adjustment (fitted scale) --
reporting both honestly rather than assuming the fancier one wins.

Win probability uses the closed-form Skellam distribution (exact, no
sampling needed) for speed across thousands of backtest games; live
predictions use actual Monte Carlo sampling to also get totals/blowout
probability, which have no simple closed form.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
from scipy.stats import skellam

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.simulation import build_scoring_rates
from pipeline.mlb.pitcher_ratings import build_sp_ratings, build_bullpen_ratings
from pipeline.common.metrics import brier_score, log_loss, accuracy

TRAIN_SEASONS = [2024, 2025]  # matches pitcher-data availability window from Phase 1
TEST_SEASONS = [2026]
EXTRA_INNINGS_HOME_EDGE = 0.52


def skellam_home_win_prob(lam_home, lam_away):
    lam_home = np.clip(lam_home, 0.3, None)
    lam_away = np.clip(lam_away, 0.3, None)
    p_home_gt = skellam.sf(0, lam_home, lam_away)
    p_tie = skellam.pmf(0, lam_home, lam_away)
    return p_home_gt + p_tie * EXTRA_INNINGS_HOME_EDGE


def main():
    games = load_games()
    rates = build_scoring_rates(games)

    games = games.merge(rates.rename(columns={"team": "home_team", "off_trailing": "home_off", "def_trailing": "home_def"}),
                        on=["home_team", "game_date"], how="left")
    games = games.merge(rates.rename(columns={"team": "away_team", "off_trailing": "away_off", "def_trailing": "away_def"}),
                        on=["away_team", "game_date"], how="left")

    train = games[games["season"].isin(TRAIN_SEASONS)].dropna(subset=["home_off", "away_def", "away_off", "home_def"])
    test = games[games["season"].isin(TEST_SEASONS)].dropna(subset=["home_off", "away_def", "away_off", "home_def"])
    league_avg = float(pd.concat([train["home_score"], train["away_score"]]).mean())
    print(f"Train: {len(train)} games. Test: {len(test)} games. League avg runs/team/game: {league_avg:.3f}")

    def lambdas(df, hf_mult, sp_scale=0.0, away_sp=None, home_sp=None, away_bp=None, home_bp=None):
        lam_h = league_avg * (df["home_off"] / league_avg) * (df["away_def"] / league_avg) * hf_mult
        lam_a = league_avg * (df["away_off"] / league_avg) * (df["home_def"] / league_avg)
        if sp_scale:
            lam_h = (lam_h - sp_scale * (away_sp.fillna(0) + away_bp.fillna(0))).clip(lower=0.3)
            lam_a = (lam_a - sp_scale * (home_sp.fillna(0) + home_bp.fillna(0))).clip(lower=0.3)
        return lam_h, lam_a

    print("\nFitting home-field multiplier on train (pure offense/defense model)...")
    best = None
    for hf in np.arange(1.0, 1.16, 0.01):
        lam_h, lam_a = lambdas(train, hf)
        preds = skellam_home_win_prob(lam_h.values, lam_a.values)
        mask = train["home_win"] != 0.5
        ll = log_loss(train.loc[mask, "home_win"], preds[mask.values])
        if best is None or ll < best[0]:
            best = (ll, hf)
    _, best_hf = best
    print(f"Best home-field multiplier: {best_hf:.2f}")

    lam_h, lam_a = lambdas(test, best_hf)
    pure_preds = skellam_home_win_prob(lam_h.values, lam_a.values)

    print("Fitting starting-pitcher/bullpen adjustment scale on train...")
    sp_ratings = build_sp_ratings().drop_duplicates(["team", "game_date"], keep="first")
    bp_ratings = build_bullpen_ratings()

    def attach_pitcher(df):
        df = df.merge(sp_ratings.rename(columns={"team": "home_team", "sp_rating": "home_sp"}),
                      on=["home_team", "game_date"], how="left")
        df = df.merge(sp_ratings.rename(columns={"team": "away_team", "sp_rating": "away_sp"}),
                      on=["away_team", "game_date"], how="left")
        df = df.merge(bp_ratings.rename(columns={"team": "home_team", "bullpen_rating": "home_bp"}),
                      on=["home_team", "game_date"], how="left")
        df = df.merge(bp_ratings.rename(columns={"team": "away_team", "bullpen_rating": "away_bp"}),
                      on=["away_team", "game_date"], how="left")
        return df

    train_p = attach_pitcher(train)
    test_p = attach_pitcher(test)

    best_scale = None
    for scale in np.arange(0.0, 0.31, 0.02):
        lam_h, lam_a = lambdas(train_p, best_hf, sp_scale=scale,
                               away_sp=train_p["away_sp"], home_sp=train_p["home_sp"],
                               away_bp=train_p["away_bp"], home_bp=train_p["home_bp"])
        preds = skellam_home_win_prob(lam_h.values, lam_a.values)
        mask = train_p["home_win"] != 0.5
        ll = log_loss(train_p.loc[mask, "home_win"], preds[mask.values])
        if best_scale is None or ll < best_scale[0]:
            best_scale = (ll, scale)
    _, sp_scale = best_scale
    print(f"Best SP/bullpen adjustment scale: {sp_scale:.2f}")

    lam_h, lam_a = lambdas(test_p, best_hf, sp_scale=sp_scale,
                           away_sp=test_p["away_sp"], home_sp=test_p["home_sp"],
                           away_bp=test_p["away_bp"], home_bp=test_p["home_bp"])
    adjusted_preds = skellam_home_win_prob(lam_h.values, lam_a.values)

    y_test = test["home_win"].values
    mask = y_test != 0.5

    with open(ROOT / "notebooks_out" / "mlb_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]
    with open(ROOT / "notebooks_out" / "mlb_pitcher_model_backtest.json") as f:
        blend = json.load(f)

    print(f"\n=== Out-of-sample results, {TEST_SEASONS} (n={mask.sum()}) ===")
    print(f"elo_only (existing baseline):        Brier={blend['elo_only']['brier']:.4f}  LogLoss={blend['elo_only']['log_loss']:.4f}  Acc={blend['elo_only']['accuracy']:.4f}")
    print(f"elo+SP+bullpen (existing, deployed): Brier={blend['blend']['brier']:.4f}  LogLoss={blend['blend']['log_loss']:.4f}  Acc={blend['blend']['accuracy']:.4f}")
    print(f"simulation, pure off/def:            Brier={brier_score(y_test[mask], pure_preds[mask]):.4f}  LogLoss={log_loss(y_test[mask], pure_preds[mask]):.4f}  Acc={accuracy(y_test[mask], pure_preds[mask]):.4f}")
    print(f"simulation, +SP/bullpen adjustment:  Brier={brier_score(y_test[mask], adjusted_preds[mask]):.4f}  LogLoss={log_loss(y_test[mask], adjusted_preds[mask]):.4f}  Acc={accuracy(y_test[mask], adjusted_preds[mask]):.4f}")

    out = {
        "home_field_mult": float(best_hf), "sp_bullpen_scale": float(sp_scale), "league_avg_runs": league_avg,
        "pure": {"brier": brier_score(y_test[mask], pure_preds[mask]), "log_loss": log_loss(y_test[mask], pure_preds[mask]), "accuracy": accuracy(y_test[mask], pure_preds[mask])},
        "adjusted": {"brier": brier_score(y_test[mask], adjusted_preds[mask]), "log_loss": log_loss(y_test[mask], adjusted_preds[mask]), "accuracy": accuracy(y_test[mask], adjusted_preds[mask])},
    }
    with open(ROOT / "notebooks_out" / "mlb_simulation_backtest.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to notebooks_out/mlb_simulation_backtest.json")


if __name__ == "__main__":
    main()

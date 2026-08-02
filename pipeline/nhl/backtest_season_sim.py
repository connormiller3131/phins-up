"""Does the season simulator (pipeline/common/season_sim.py) produce a
well-calibrated projection for NHL? Same real-data-only methodology as
pipeline/mlb/backtest_season_sim.py and pipeline/nfl/backtest_season_sim.py:
freeze a snapshot partway through several real, fully-completed past
seasons, simulate the rest from real Elo ratings, and check simulated
division-title rates against what actually happened.

data/nhl/team_games.parquet only has completed real games (pulled from the
league schedule endpoint filtered to finished game states) -- perfectly
sufficient for backtesting past, fully-over seasons, but NOT what the live
production wrapper will use for the remaining schedule (that needs the same
live /v1/schedule endpoint pull generate_current_slate.py already uses for
upcoming games, since future games aren't in this historical-only file)."""
import sys
import pathlib
import numpy as np
import pandas as pd
import polars as pl
import json
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nhl.elo_model import run_elo
from pipeline.nhl.team_map import normalize_team
from pipeline.common.season_sim import simulate_remaining_wins, division_title_rates, shrink_toward_field
from pipeline.common.metrics import brier_score, calibration_curve

DATA_DIR = ROOT / "data" / "nhl"
BACKTEST_SEASONS = [2021, 2022, 2023, 2024, 2025]  # season = start year (2021 -> 2021-22 season)
SNAPSHOT_MONTH_DAYS = [("12", "15"), ("02", "01")]  # ~25% and ~55% through an Oct-Apr season

DIVISIONS = {}


def _load_divisions():
    global DIVISIONS
    if DIVISIONS:
        return DIVISIONS
    resp = requests.get("https://api-web.nhle.com/v1/standings/now", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    DIVISIONS = {normalize_team(t["teamAbbrev"]["default"]): t["divisionName"] for t in data["standings"]}
    return DIVISIONS


def _season_games(season):
    df = pl.read_parquet(DATA_DIR / "team_games.parquet").to_pandas()
    df = df[df["season"] == season].copy()
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df.sort_values("game_date").reset_index(drop=True)


def _games_through(max_season):
    """Every real completed game through and including max_season -- used
    only for the rating computation, which (matching production) needs Elo
    carried forward across real season boundaries via season_regression,
    not re-initialized to a blank 1500 at the start of the backtested
    season."""
    df = pl.read_parquet(DATA_DIR / "team_games.parquet").to_pandas()
    df = df[df["season"] <= max_season].copy()
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df.sort_values("game_date").reset_index(drop=True)


def _win_counts(rows):
    wins = {}
    for _, r in rows.iterrows():
        wins.setdefault(r["home_team"], 0)
        wins.setdefault(r["away_team"], 0)
        if r["home_win"] == 1.0:
            wins[r["home_team"]] += 1
        else:
            wins[r["away_team"]] += 1
    return wins


def run_one_snapshot(season, snapshot_date, elo_params, n_sims=3000):
    games = _season_games(season)
    before = games[games["game_date"] < snapshot_date].reset_index(drop=True)
    after = games[games["game_date"] >= snapshot_date].reset_index(drop=True)
    if len(before) < 150 or len(after) < 150:
        return None

    history = _games_through(season)
    history_before = history[history["game_date"] < snapshot_date].reset_index(drop=True)
    _, ratings = run_elo(history_before, k=elo_params["k"], home_adv=elo_params["home_adv"],
                          scale=elo_params["scale"], season_regression=elo_params["season_regression"],
                          return_ratings=True)
    current_wins = _win_counts(before)
    remaining_games = after[["home_team", "away_team"]]
    teams, sim_wins = simulate_remaining_wins(remaining_games, ratings, elo_params["home_adv"], elo_params["scale"], n_sims=n_sims)

    base_wins = np.array([current_wins.get(t, 0) for t in teams])
    final_sim_wins = base_wins[None, :] + sim_wins
    real_final_wins = _win_counts(games)

    sim_mean = final_sim_wins.mean(axis=0)
    real_arr = np.array([real_final_wins.get(t, 0) for t in teams])
    rmse = float(np.sqrt(np.mean((sim_mean - real_arr) ** 2)))

    games_played_before = {}
    for _, r in before.iterrows():
        games_played_before[r["home_team"]] = games_played_before.get(r["home_team"], 0) + 1
        games_played_before[r["away_team"]] = games_played_before.get(r["away_team"], 0) + 1
    remaining_game_counts = {}
    for _, r in after.iterrows():
        remaining_game_counts[r["home_team"]] = remaining_game_counts.get(r["home_team"], 0) + 1
        remaining_game_counts[r["away_team"]] = remaining_game_counts.get(r["away_team"], 0) + 1
    naive_final = []
    for t in teams:
        gp = games_played_before.get(t, 0)
        win_rate = current_wins.get(t, 0) / gp if gp else 0.5
        naive_final.append(current_wins.get(t, 0) + win_rate * remaining_game_counts.get(t, 0))
    naive_rmse = float(np.sqrt(np.mean((np.array(naive_final) - real_arr) ** 2)))

    divisions = _load_divisions()
    team_divisions = {t: divisions[t] for t in teams if t in divisions}
    rates = division_title_rates(teams, final_sim_wins, team_divisions)

    obs_probs, obs_outcomes = [], []
    for div, team_rates in rates.items():
        div_teams = [t for t, d in team_divisions.items() if d == div]
        real_winner = max(div_teams, key=lambda t: real_final_wins.get(t, 0))
        for t, p in team_rates.items():
            obs_probs.append(p)
            obs_outcomes.append(1.0 if t == real_winner else 0.0)

    return {"season": season, "snapshot": str(snapshot_date.date()), "rmse": rmse, "naive_rmse": naive_rmse,
            "obs_probs": obs_probs, "obs_outcomes": obs_outcomes}


def main():
    elo_params = json.load(open(ROOT / "notebooks_out" / "nhl_win_prob_backtest.json"))["elo_params"]

    all_results = []
    for season in BACKTEST_SEASONS:
        for month, day in SNAPSHOT_MONTH_DAYS:
            year = season + 1 if month in ("01", "02") else season
            snapshot_date = pd.Timestamp(f"{year}-{month}-{day}")
            r = run_one_snapshot(season, snapshot_date, elo_params)
            if r is None:
                print(f"{season} {snapshot_date.date()}: skipped")
                continue
            all_results.append(r)
            print(f"{season} snapshot {r['snapshot']}: RMSE={r['rmse']:.2f} (naive={r['naive_rmse']:.2f}) "
                  f"{'BEATS' if r['rmse'] < r['naive_rmse'] else 'LOSES TO'} naive baseline")

    n_beats = sum(1 for r in all_results if r["rmse"] < r["naive_rmse"])
    print(f"\nBeats naive baseline in {n_beats}/{len(all_results)} snapshots.")

    all_probs = np.concatenate([r["obs_probs"] for r in all_results])
    all_outcomes = np.concatenate([r["obs_outcomes"] for r in all_results])
    n_teams = 8  # every current NHL division has 8 teams
    naive_brier = (1 - 1/n_teams)**2 * (1/n_teams) + (1/n_teams)**2 * (1 - 1/n_teams)
    print(f"\nPooled division-title calibration across {len(all_results)} snapshots, {len(all_probs)} observations:")
    print(f"  Brier score: {brier_score(all_outcomes, all_probs):.4f}  (naive equal-probability baseline: {naive_brier:.4f})")
    print("\n  Calibration curve:")
    for row in calibration_curve(all_outcomes, all_probs, n_bins=5):
        print(f"    [{row['bin_lo']:.1f}-{row['bin_hi']:.1f}]: predicted avg {row['predicted_mean']:.3f}, "
              f"actual {row['actual_mean']:.3f}, n={row['count']}")

    print("\n  Shrinkage-factor sweep (toward the field's naive 1/8 baseline):")
    best = None
    for lam in np.linspace(0, 1, 21):
        b = brier_score(all_outcomes, shrink_toward_field(all_probs, n_teams, lam))
        if best is None or b < best[0]:
            best = (b, lam)
        print(f"    lambda={lam:.2f}  brier={b:.4f}")
    print(f"\n  best lambda: {best[1]:.2f}  brier: {best[0]:.4f}")


if __name__ == "__main__":
    main()

"""Does the season simulator (pipeline/common/season_sim.py) produce a
well-calibrated projection for NFL? Same real-data-only "prove it before
shipping" methodology as pipeline/mlb/backtest_season_sim.py: freeze a
snapshot partway through several real, fully-completed past regular seasons,
simulate the rest from real Elo ratings, and check simulated division-title
rates against what actually happened -- no peeking at games after the
snapshot. NFL has far fewer games per season than MLB (17 vs 162), so this
pools more seasons rather than more snapshots per season to get a reasonable
sample."""
import sys
import pathlib
import numpy as np
import polars as pl
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.elo_model import run_elo
from pipeline.common.season_sim import simulate_remaining_wins, division_title_rates, shrink_toward_field
from pipeline.common.metrics import brier_score, calibration_curve

DATA_DIR = ROOT / "data" / "nfl"
BACKTEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
SNAPSHOT_WEEKS = [6, 12]  # ~35% and ~70% through a real 17-game season

# Real divisional alignment (nflreadpy's team_conf/team_division, confirmed
# via data/nfl/teams.parquet -- same source Phase 1's standings uses).
DIVISIONS = {}


def _load_divisions():
    global DIVISIONS
    if DIVISIONS:
        return DIVISIONS
    teams = pl.read_parquet(DATA_DIR / "teams.parquet").to_pandas()
    teams = teams.drop_duplicates("team_abbr")
    DIVISIONS = dict(zip(teams["team_abbr"], teams["team_division"]))
    return DIVISIONS


def _season_games(season):
    raw = pl.read_parquet(DATA_DIR / "schedules.parquet")
    reg = raw.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    df = reg.select(["season", "week", "home_team", "away_team", "home_score", "away_score"]).to_pandas()
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df.sort_values("week").reset_index(drop=True)


def _reg_games_through(max_season):
    """Every real completed regular-season game up through and including
    max_season -- used only for the rating computation, which (matching
    production) needs Elo carried forward across real season boundaries via
    season_regression, not re-initialized to a blank 1500 at the start of
    the season being backtested."""
    raw = pl.read_parquet(DATA_DIR / "schedules.parquet")
    reg = raw.filter((pl.col("season") <= max_season) & (pl.col("game_type") == "REG")
                     & pl.col("home_score").is_not_null())
    df = reg.select(["season", "week", "home_team", "away_team", "home_score", "away_score"]).to_pandas()
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df.sort_values(["season", "week"]).reset_index(drop=True)


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


def run_one_snapshot(season, snapshot_week, elo_params, n_sims=3000):
    games = _season_games(season)
    before = games[games["week"] < snapshot_week].reset_index(drop=True)
    after = games[games["week"] >= snapshot_week].reset_index(drop=True)
    if len(before) < 30 or len(after) < 30:
        return None

    # Ratings from full history through this snapshot (all prior seasons in
    # full, plus this season's real games so far) -- matches production
    # (build_nfl_title_odds), not just this one season in isolation.
    history = _reg_games_through(season)
    history_before = history[(history["season"] < season) | ((history["season"] == season) & (history["week"] < snapshot_week))].reset_index(drop=True)
    _, ratings = run_elo(history_before, k=elo_params["k"], home_adv=elo_params["home_adv"],
                          scale=elo_params["scale"], rest_adv=elo_params["rest_adv"],
                          season_regression=elo_params["season_regression"], return_ratings=True)
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

    return {"season": season, "week": snapshot_week, "rmse": rmse, "naive_rmse": naive_rmse,
            "obs_probs": obs_probs, "obs_outcomes": obs_outcomes}


def main():
    elo_params = json.load(open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json"))["elo_params"]

    all_results = []
    for season in BACKTEST_SEASONS:
        for wk in SNAPSHOT_WEEKS:
            r = run_one_snapshot(season, wk, elo_params)
            if r is None:
                print(f"{season} week {wk}: skipped")
                continue
            all_results.append(r)
            print(f"{season} week {wk}: RMSE={r['rmse']:.2f} (naive={r['naive_rmse']:.2f}) "
                  f"{'BEATS' if r['rmse'] < r['naive_rmse'] else 'LOSES TO'} naive baseline")

    n_beats = sum(1 for r in all_results if r["rmse"] < r["naive_rmse"])
    print(f"\nBeats naive baseline in {n_beats}/{len(all_results)} snapshots.")

    all_probs = np.concatenate([r["obs_probs"] for r in all_results])
    all_outcomes = np.concatenate([r["obs_outcomes"] for r in all_results])
    n_teams = 4  # every NFL division has 4 teams
    naive_brier = (1 - 1/n_teams)**2 * (1/n_teams) + (1/n_teams)**2 * (1 - 1/n_teams)
    print(f"\nPooled division-title calibration across {len(all_results)} snapshots, {len(all_probs)} observations:")
    print(f"  Brier score: {brier_score(all_outcomes, all_probs):.4f}  (naive equal-probability baseline: {naive_brier:.4f})")
    print("\n  Calibration curve:")
    for row in calibration_curve(all_outcomes, all_probs, n_bins=5):
        print(f"    [{row['bin_lo']:.1f}-{row['bin_hi']:.1f}]: predicted avg {row['predicted_mean']:.3f}, "
              f"actual {row['actual_mean']:.3f}, n={row['count']}")

    print("\n  Shrinkage-factor sweep (toward the field's naive 1/4 baseline):")
    best = None
    for lam in np.linspace(0, 1, 21):
        b = brier_score(all_outcomes, shrink_toward_field(all_probs, n_teams, lam))
        if best is None or b < best[0]:
            best = (b, lam)
        print(f"    lambda={lam:.2f}  brier={b:.4f}")
    print(f"\n  best lambda: {best[1]:.2f}  brier: {best[0]:.4f}")


if __name__ == "__main__":
    main()

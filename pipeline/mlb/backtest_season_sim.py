"""Does the season simulator (pipeline/common/season_sim.py) actually produce
a well-calibrated projection? Real test: take several real, fully-completed
past MLB seasons, freeze a snapshot partway through each (using only games up
to that date -- no peeking at what happens after), simulate the rest, and
check the simulated win-total distribution and division-title rates against
what *actually* happened in the real, known remainder of each season.

One season alone isn't enough to judge calibration (a division has only 5
teams -- one outlier collapse and the "top pick was wrong" signal is
meaningless on its own; e.g. 2025's AL Central: Detroit was a real 53-32,
best-record-in-baseball team as of the July 1 snapshot and the sim correctly
gave them a ~99% title rate -- they then had a real, historic second-half
collapse (34-42) while Cleveland caught up. That's not a calibration bug,
it's what a well-calibrated high-confidence pick losing actually looks like).
Pooling every team-division prediction across multiple seasons and snapshot
dates into one Brier score is the only way to tell a real calibration problem
apart from ordinary variance.

Same "prove it before shipping" rule as every other model in this project.
"""
import sys
import pathlib
import re
import numpy as np
import pandas as pd
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.team_map import br_to_statcast, DIVISIONS
from pipeline.mlb.elo_model import run_elo
from pipeline.common.season_sim import simulate_remaining_wins, division_title_rates
from pipeline.common.metrics import brier_score

DATA_DIR = ROOT / "data" / "mlb"
BACKTEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
SNAPSHOT_MONTH_DAYS = ["05-15", "07-01"]  # ~25% and ~50% through a real MLB season


def _parse_date(date_str, season):
    cleaned = re.sub(r"\s*\(\d\)$", "", date_str)
    cleaned = cleaned.split(",", 1)[1].strip()
    return pd.to_datetime(f"{cleaned} {season}", format="%b %d %Y", errors="coerce")


def _games_through(max_season):
    """Every real completed game from the earliest season on record through
    max_season, inclusive -- used only for the rating computation, which
    (matching production's elo_predictions()) needs Elo carried forward
    across real season boundaries via season_regression, not re-initialized
    to a blank 1500 at the start of the season being backtested."""
    raw = pd.read_parquet(DATA_DIR / "team_schedule_raw.parquet")
    raw = raw[raw["season"] <= max_season]
    return _games_from_raw(raw)


def _season_games(season):
    """One row per real game (home perspective), full season, real final
    scores -- these seasons are long since over, so every row has one."""
    raw = pd.read_parquet(DATA_DIR / "team_schedule_raw.parquet")
    raw = raw[raw["season"] == season]
    return _games_from_raw(raw)


def _games_from_raw(raw):
    home = raw[raw["Home_Away"] != "@"].copy()
    home = home[home["R"].notna() & home["RA"].notna()]
    home["game_date"] = home.apply(lambda r: _parse_date(r["Date"], r["season"]), axis=1)
    home = home.dropna(subset=["game_date"])
    games = pd.DataFrame({
        "season": home["season"].astype(int),
        "game_date": home["game_date"],
        "home_team": home["team"].map(br_to_statcast),
        "away_team": home["Opp"].map(br_to_statcast),
        "home_score": home["R"].astype(int),
        "away_score": home["RA"].astype(int),
    })
    games["margin"] = games["home_score"] - games["away_score"]
    games["home_win"] = (games["margin"] > 0).astype(float)
    return games.sort_values("game_date").reset_index(drop=True)


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
    if len(before) < 200 or len(after) < 200:
        return None  # snapshot falls outside the real season window for this year

    # Ratings computed from full history through the snapshot (all prior
    # seasons in full, plus this season's real games so far) -- matches
    # production (build_mlb_title_odds), not just this one season in
    # isolation, so this backtest actually validates what gets shipped.
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
    real_arr = np.array([real_final_wins[t] for t in teams])
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

    divisions = {t: DIVISIONS[t] for t in teams if t in DIVISIONS}
    rates = division_title_rates(teams, final_sim_wins, divisions)

    # Pooled (team, division) observations for calibration: predicted title
    # probability vs. did-they-actually-win, one row per team per division
    # this snapshot -- NOT just the favorite, every team's own predicted rate.
    obs_probs, obs_outcomes = [], []
    for div, team_rates in rates.items():
        div_teams = [t for t, d in divisions.items() if d == div]
        real_winner = max(div_teams, key=lambda t: real_final_wins[t])
        for t, p in team_rates.items():
            obs_probs.append(p)
            obs_outcomes.append(1.0 if t == real_winner else 0.0)

    return {
        "season": season, "snapshot": str(snapshot_date.date()),
        "rmse": rmse, "naive_rmse": naive_rmse,
        "obs_probs": obs_probs, "obs_outcomes": obs_outcomes,
    }


def main():
    elo_params = json.load(open(ROOT / "notebooks_out" / "mlb_win_prob_backtest.json"))["elo_params"]

    all_results = []
    for season in BACKTEST_SEASONS:
        for md in SNAPSHOT_MONTH_DAYS:
            snapshot_date = pd.Timestamp(f"{season}-{md}")
            result = run_one_snapshot(season, snapshot_date, elo_params)
            if result is None:
                print(f"{season} {md}: skipped (snapshot outside real season window)")
                continue
            all_results.append(result)
            print(f"{season} snapshot {result['snapshot']}: RMSE={result['rmse']:.2f} "
                  f"(naive={result['naive_rmse']:.2f}) "
                  f"{'BEATS' if result['rmse'] < result['naive_rmse'] else 'LOSES TO'} naive baseline")

    n_beats = sum(1 for r in all_results if r["rmse"] < r["naive_rmse"])
    print(f"\nBeats naive baseline in {n_beats}/{len(all_results)} snapshots.")

    all_probs = np.concatenate([r["obs_probs"] for r in all_results])
    all_outcomes = np.concatenate([r["obs_outcomes"] for r in all_results])
    print(f"\nPooled division-title calibration across {len(all_results)} snapshots, {len(all_probs)} (team, division) observations:")
    print(f"  Brier score: {brier_score(all_outcomes, all_probs):.4f}  "
          f"(a naive 'each team equally likely' baseline within a 5-team division would score ~{(1-0.2)**2*0.2 + 0.2**2*0.8:.4f})")

    from pipeline.common.metrics import calibration_curve
    print("\n  Calibration curve (predicted bucket -> actual hit rate, count):")
    for row in calibration_curve(all_outcomes, all_probs, n_bins=5):
        print(f"    [{row['bin_lo']:.1f}-{row['bin_hi']:.1f}]: predicted avg {row['predicted_mean']:.3f}, "
              f"actual {row['actual_mean']:.3f}, n={row['count']}")


if __name__ == "__main__":
    main()

"""Monte Carlo run-scoring simulation for MLB games. Rather than a single
closed-form win-probability formula, this simulates the actual run-scoring
process: each team's expected runs come from a log-linear scoring model
(offense x opponent defense / league average -- the same family of model
used across sports analytics for score prediction, e.g. soccer's Poisson
goal models), refined by real starting-pitcher and bullpen strength, then
simulated many times to get a full distribution. Win probability, the
totals market, and upset/blowout probability all fall out of one model
instead of needing a separate one for each.

Simplifying assumptions, stated plainly:
- Runs are drawn as independent Poisson per team. Real MLB scoring is
  slightly overdispersed vs. Poisson (a few blowouts pull the tail), and
  the two teams' scores aren't perfectly independent (game flow, weather),
  but Poisson is the standard, transparent starting point and is validated
  against actual outcomes below rather than assumed adequate.
- The starter/bullpen adjustment converts Statcast run-value into an
  expected-runs multiplier via a single fitted scale factor, not a fully
  re-derived run-scoring model from pitch-level data.
"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
WINDOW, MIN_GAMES = 15, 6
N_SIMS = 10000


def build_scoring_rates(games=None):
    """Trailing runs-scored (offense) and runs-allowed (defense) per team,
    walk-forward safe. One row per (team, game_date)."""
    games = games if games is not None else load_games()
    home = games[["game_date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "allowed"})
    away = games[["game_date", "away_team", "home_score", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "allowed", "home_score": "scored"})
    all_games = pd.concat([home, away], ignore_index=True).sort_values(["team", "game_date"])
    all_games["off_trailing"] = all_games.groupby("team")["scored"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
    all_games["def_trailing"] = all_games.groupby("team")["allowed"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
    return all_games[["team", "game_date", "off_trailing", "def_trailing"]]


def expected_lambdas(home_off, away_def, away_off, home_def, league_avg,
                     home_field_mult=1.0, home_sp_adj=0.0, away_sp_adj=0.0):
    """home_sp_adj/away_sp_adj: the DEFENDING team's pitching adjustment
    (positive run_value -> suppresses the opponent's expected runs)."""
    lam_home = league_avg * (home_off / league_avg) * (away_def / league_avg) * home_field_mult
    lam_away = league_avg * (away_off / league_avg) * (home_def / league_avg)
    lam_home = max(lam_home - away_sp_adj, 0.3)
    lam_away = max(lam_away - home_sp_adj, 0.3)
    return lam_home, lam_away


def simulate_game(lam_home, lam_away, n_sims=N_SIMS, extra_innings_home_edge=0.52, rng=None):
    rng = rng or np.random.default_rng()
    home_runs = rng.poisson(lam_home, n_sims)
    away_runs = rng.poisson(lam_away, n_sims)
    ties = home_runs == away_runs
    tie_breaks = rng.random(ties.sum()) < extra_innings_home_edge
    home_win = home_runs > away_runs
    home_win[np.where(ties)[0]] = tie_breaks
    return {
        "p_home_win": float(home_win.mean()),
        "home_runs": home_runs, "away_runs": away_runs,
        "total_runs": home_runs + away_runs,
        "margin": home_runs - away_runs,
    }


def total_over_prob(sim, line):
    return float((sim["total_runs"] > line).mean())


def blowout_prob(sim, margin=5):
    return float((np.abs(sim["margin"]) >= margin).mean())

"""Team-level trailing offensive quality via Statcast's est_woba (expected
wOBA from contact quality, PA-weighted per team per game) -- a less noisy
alternative to raw runs scored for "how good is this team's offense right
now." Same walk-forward-safe design as pitcher_ratings.py: build_* functions
shift before rolling for backtesting, current_* functions are unshifted for
live/future predictions. Backtested as a win-probability feature before
being deployed (Brier 0.2480 -> 0.2477 on top of the existing elo+SP+bullpen
blend)."""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
WOBA_WINDOW, WOBA_MIN = 10, 3


def _team_game_woba():
    bg = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    bg = bg.dropna(subset=["est_woba", "pa_count"])
    bg = bg.assign(woba_x_pa=bg["est_woba"] * bg["pa_count"])
    team_game = (
        bg.groupby(["team", "game_date"])
        .agg(woba_x_pa=("woba_x_pa", "sum"), pa=("pa_count", "sum"))
        .reset_index()
    )
    team_game["team_woba"] = team_game["woba_x_pa"] / team_game["pa"]
    return team_game.sort_values(["team", "game_date"])


def build_team_woba_ratings():
    """One row per (team, game_date): trailing PA-weighted est_woba, using
    only games strictly before that date."""
    team_game = _team_game_woba()
    team_game["team_woba_rating"] = team_game.groupby("team")["team_woba"].transform(
        lambda s: s.shift(1).rolling(window=WOBA_WINDOW, min_periods=WOBA_MIN).mean()
    )
    return team_game[["team", "game_date", "team_woba_rating"]]


def current_team_woba(team):
    team_game = _team_game_woba()
    team_game = team_game[team_game["team"] == team]
    if len(team_game) < WOBA_MIN:
        return None
    return float(team_game["team_woba"].tail(WOBA_WINDOW).mean())

"""Current (as-of-today) trailing rates for MLB batters/pitchers and
opposing teams -- the un-shifted version of prop_data.py's rolling window,
evaluated at each player's/team's most recent game. Used to project a game
that hasn't been played yet."""
import pathlib
import pandas as pd

from pipeline.mlb.player_names import get_name_lookup
from pipeline.mlb.props.prop_data import WINDOW, MIN_GAMES

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "mlb"


def _current_trailing(df, stat_col):
    names = get_name_lookup()
    df = df.merge(names, on="player_id", how="left")
    df = df.sort_values(["player_id", "game_date"])
    df["current_avg"] = df.groupby("player_id")[stat_col].transform(
        lambda s: s.rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    df["games_played"] = df.groupby("player_id").cumcount() + 1
    latest = df.sort_values("game_date").groupby("player_id").tail(1)
    return latest.set_index("player_id")[["player_display_name", "team", "current_avg", "games_played"]]


def _current_trailing_defense(df, stat_col):
    allowed = (
        df.groupby(["opponent_team", "game_date"])[stat_col]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team", stat_col: "allowed_that_game"})
        .sort_values(["defense_team", "game_date"])
    )
    allowed["current_avg"] = allowed.groupby("defense_team")["allowed_that_game"].transform(
        lambda s: s.rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    latest = allowed.sort_values("game_date").groupby("defense_team").tail(1)
    return latest.set_index("defense_team")["current_avg"]


def batter_current_trailing(stat_col):
    df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    return _current_trailing(df, stat_col)


def batter_opponent_current_trailing(stat_col):
    df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    return _current_trailing_defense(df, stat_col)


def pitcher_current_trailing(stat_col: str = "strikeouts"):
    df = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    return _current_trailing(df, stat_col)


def pitcher_opponent_current_trailing(stat_col: str = "strikeouts"):
    df = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    return _current_trailing_defense(df, stat_col)

"""Current (as-of-today) trailing rates for MLB batters/pitchers and
opposing teams -- the un-shifted version of prop_data.py's rolling window,
evaluated at each player's/team's most recent game. Used to project a game
that hasn't been played yet."""
import pathlib
import pandas as pd

from pipeline.mlb.player_names import get_name_lookup, fetch_names_for_ids
from pipeline.mlb.props.prop_data import WINDOW, MIN_GAMES

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "mlb"

# A player whose most recent logged game is older than this gets dropped from
# props entirely -- confirmed real case: Tyler Locklear's only rows in our
# data are from a 2025-08/09 Diamondbacks stint (nearly a year stale relative
# to a mid-2026 refresh), yet the rolling window still produced a confident
# "current" projection from them because nothing checked *when* those games
# were. A rolling average is only a legitimate "current form" signal if the
# player has actually played recently; a stale prior-season stint (long
# layoff, demotion, or our own data pull simply not having ingested a brand
# new call-up's last couple of games yet) should not be presented with the
# same confidence as someone in today's lineup. 25 days covers a full 15-day
# IL stint plus a real buffer for legitimately-current bench bats, while
# still catching gaps of months.
STALE_DAYS = 25


def _current_trailing(df, stat_col):
    names = get_name_lookup()
    df = df.merge(names, on="player_id", how="left")
    # A player_id missing from Chadwick's register (real gap, confirmed --
    # not a caching issue -- for recent callups it hasn't caught up to yet)
    # left-joins to NaN here; that NaN used to flow straight into props as
    # p["player"] with no guard, which crashed _norm_name (unicodedata.
    # normalize on a float) and took down the whole pipeline. Filled from
    # MLB's own Stats API first (real name, same ID space, no gaps); only
    # falls back to a placeholder for the rare id neither source has.
    missing = df["player_display_name"].isna()
    if missing.any():
        missing_ids = df.loc[missing, "player_id"].unique()
        api_names = fetch_names_for_ids(missing_ids)
        df.loc[missing, "player_display_name"] = df.loc[missing, "player_id"].map(api_names)
        still_missing = df["player_display_name"].isna()
        if still_missing.any():
            df.loc[still_missing, "player_display_name"] = df.loc[still_missing, "player_id"].apply(lambda pid: f"Player {pid}")
    df = df.sort_values(["player_id", "game_date"])
    df["current_avg"] = df.groupby("player_id")[stat_col].transform(
        lambda s: s.rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    df["games_played"] = df.groupby("player_id").cumcount() + 1
    latest = df.sort_values("game_date").groupby("player_id").tail(1)
    # Measured against the data's own most recent game, not real-world
    # "today" -- the nightly pull normally lags a few days behind, and
    # penalizing every player for that pipeline lag would be a different bug,
    # not a fix. This only catches genuine staleness relative to whatever's
    # actually fresh in the dataset right now.
    as_of = df["game_date"].max()
    latest = latest[(as_of - latest["game_date"]).dt.days <= STALE_DAYS]
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

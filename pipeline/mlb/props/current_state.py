"""Current (as-of-today) trailing rates for MLB batters/pitchers and
opposing teams -- the un-shifted version of prop_data.py's rolling window,
evaluated at each player's/team's most recent game. Used to project a game
that hasn't been played yet."""
import pathlib
import pandas as pd

from pipeline.mlb.player_names import get_name_lookup, fetch_names_for_ids
from pipeline.mlb.props.prop_data import WINDOW, MIN_GAMES

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "mlb"

# A gap longer than this between a player's own consecutive logged games
# starts a new "stint" (see below); a player whose current stint's last game
# is older than this relative to the freshest game in the whole dataset gets
# dropped entirely. 25 days covers a full 15-day IL stint plus a real buffer
# for legitimately-current bench bats, while still catching gaps of months.
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

    # The rolling window below counts *games*, not days -- with no gap
    # awareness, a player's last WINDOW logged rows can silently span a
    # demotion or a return from a stint over a year earlier. Confirmed real
    # case: Tyler Locklear has 2 genuine 2026 games in our data, but the
    # rolling window pooled them with games from his 2025-08/09 Diamondbacks
    # stint to clear MIN_GAMES, producing a confident "current form" number
    # built mostly from year-old at-bats -- the previous fix here (a plain
    # "is the latest game stale" check) missed this because his latest game
    # genuinely isn't stale; the problem is what got blended behind it.
    # Segmenting on gaps > STALE_DAYS between a player's own consecutive
    # games isolates just their current stint (this also naturally resets at
    # every real season boundary, since that gap always exceeds STALE_DAYS),
    # so MIN_GAMES requires genuine recent volume, not games borrowed from a
    # different stint.
    gap_days = df.groupby("player_id")["game_date"].diff().dt.days
    df["new_stint"] = gap_days.isna() | (gap_days > STALE_DAYS)
    df["stint_id"] = df.groupby("player_id")["new_stint"].cumsum()
    last_stint_id = df.groupby("player_id")["stint_id"].transform("max")
    df = df[df["stint_id"] == last_stint_id]

    df["current_avg"] = df.groupby("player_id")[stat_col].transform(
        lambda s: s.rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    df["games_played"] = df.groupby("player_id").cumcount() + 1
    latest = df.sort_values("game_date").groupby("player_id").tail(1)
    # Belt-and-suspenders: also drop anyone whose current stint's last game
    # is itself stale relative to the freshest game in the whole dataset --
    # covers someone who simply hasn't played at all recently (e.g. a
    # current, undeclared injury), where there's no newer stint to speak of.
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

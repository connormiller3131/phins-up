"""Batter-vs-pitching-style trailing feature: how has this batter performed
against pitchers whose stuff resembles today's starter (fastball/sinker/
slider/curve/changeup/cutter mix + velocity), not just same-handed pitchers.
Reuses batter_game_logs game totals directly, attributing each game to
whichever pitcher started for the opponent that day (the same approximation
already used for is_starter elsewhere in this project) rather than
re-deriving a new PA-level pull. Walk-forward safe.
"""
import pathlib
import pandas as pd

from pipeline.mlb.pitcher_style_clusters import build_clusters
from pipeline.mlb.props.hand_features import starters_by_team_date

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "mlb"
WINDOW, MIN_GAMES = 20, 6


def attach_style_feature(df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """df must have columns: player_id, opponent_team, game_date, and
    (importantly) `actual` already set to stat_col's values for THIS df's
    own rows -- but since we need the batter's OWN historical values by
    style cluster, we rebuild from the full batter_game_logs, not from df,
    so this works even if df has been filtered/renamed upstream."""
    style = build_clusters()[["style_cluster"]]
    starters = starters_by_team_date()
    starters = starters.merge(style, left_on="starter_id", right_index=True, how="left")

    df = df.merge(starters.rename(columns={"team": "opponent_team"}),
                  on=["opponent_team", "game_date"], how="left")

    full_batter_games = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    full_batter_games = full_batter_games.merge(
        starters.rename(columns={"team": "opponent_team"}), on=["opponent_team", "game_date"], how="left")
    full_batter_games = full_batter_games.dropna(subset=["style_cluster"]).sort_values(
        ["player_id", "style_cluster", "game_date"])
    full_batter_games["trailing"] = full_batter_games.groupby(["player_id", "style_cluster"])[stat_col].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
    )
    style_trail = full_batter_games[["player_id", "style_cluster", "game_date", "trailing"]].rename(
        columns={"trailing": "own_vs_style_trailing_avg"}
    ).sort_values("game_date")

    resolved = df.dropna(subset=["style_cluster"]).sort_values("game_date")
    merged = pd.merge_asof(
        resolved, style_trail, on="game_date", by=["player_id", "style_cluster"], direction="backward",
    )
    unresolved = df[df["style_cluster"].isna()].copy()
    unresolved["own_vs_style_trailing_avg"] = pd.NA

    out = pd.concat([merged, unresolved], ignore_index=True).drop(columns=["starter_id", "style_cluster"])
    return out

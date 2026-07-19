"""Trailing starting-pitcher and bullpen quality ratings, built from
Statcast's delta_pitcher_run_exp -- a context-neutral, per-pitch run-value
metric (positive = good for the pitcher; empirically confirmed: a home run
allowed is about -1.7, a strikeout about +0.22). This is a materially
better signal than ERA/runs-allowed over small samples because it isn't
distorted by bullpen luck, defense, or BABIP variance the way traditional
runs-allowed is.

All backtest-facing functions are walk-forward safe (shift(1) before the
rolling window -- a rating never reflects the game it's about to predict).
The *_current_* functions are unshifted, for live/future predictions.
"""
import pathlib
import pandas as pd

from pipeline.mlb.player_names import get_name_lookup

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"

SP_WINDOW, SP_MIN = 6, 3     # trailing starts
BP_WINDOW, BP_MIN = 10, 5    # trailing relief appearances (team-level)
BP_ARM_LOOKBACK = 15         # trailing relief appearances per pitcher, for individual arm stats


def _load_pitchers():
    return pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")


def build_sp_ratings():
    """One row per (player_id, team, game_date) for STARTERS: trailing avg
    run_value per start, using only starts strictly before that date."""
    df = _load_pitchers()
    starters = df[df["is_starter"]].sort_values(["player_id", "game_date"]).copy()
    starters["sp_rating"] = starters.groupby("player_id")["run_value"].transform(
        lambda s: s.shift(1).rolling(SP_WINDOW, min_periods=SP_MIN).mean()
    )
    return starters[["player_id", "team", "game_date", "sp_rating"]]


def build_bullpen_ratings():
    """One row per (team, game_date): trailing avg run_value per relief
    appearance for that TEAM's bullpen collectively (composition varies
    game to game, so this is tracked at the team level, not per pitcher)."""
    df = _load_pitchers()
    bp = df[~df["is_starter"]].copy()
    team_game = (
        bp.groupby(["team", "game_date"])["run_value"]
        .mean()
        .reset_index()
        .sort_values(["team", "game_date"])
    )
    team_game["bullpen_rating"] = team_game.groupby("team")["run_value"].transform(
        lambda s: s.shift(1).rolling(BP_WINDOW, min_periods=BP_MIN).mean()
    )
    return team_game[["team", "game_date", "bullpen_rating"]]


def current_sp_rating(pitcher_id):
    df = _load_pitchers()
    starts = df[(df["is_starter"]) & (df["player_id"] == pitcher_id)].sort_values("game_date")
    if len(starts) < SP_MIN:
        return None
    return float(starts["run_value"].tail(SP_WINDOW).mean())


def current_bullpen_rating(team):
    df = _load_pitchers()
    bp = df[(~df["is_starter"]) & (df["team"] == team)].sort_values("game_date")
    team_game = bp.groupby("game_date")["run_value"].mean().sort_index()
    if len(team_game) < BP_MIN:
        return None
    return float(team_game.tail(BP_WINDOW).mean())


def current_bullpen_arms(team, n=4):
    """The team's most-frequently-used relief pitchers recently (a real,
    ranked-by-actual-usage proxy for "who's likely to appear" -- there's no
    "probable reliever" the way there's a probable starter, since bullpen
    usage is a live, matchup-driven decision made during the game), each
    with their own trailing counting stats. Informational only -- no lines,
    no odds, just real recent numbers, since there's no sound basis to
    project a betting line for someone who may not even appear."""
    df = _load_pitchers()
    bp = df[(~df["is_starter"]) & (df["team"] == team)].sort_values("game_date")
    if bp.empty:
        return []

    # Rank arms by recent appearance count (most-used = most likely to see
    # the mound again), using only each pitcher's own trailing appearances.
    recent = bp.groupby("player_id").tail(BP_ARM_LOOKBACK)
    counts = bp["player_id"].value_counts()
    top_ids = counts.head(n).index.tolist()

    names = get_name_lookup()
    out = []
    for pid in top_ids:
        arm = recent[recent["player_id"] == pid].tail(BP_ARM_LOOKBACK)
        name_row = names[names["player_id"] == pid]
        name = name_row["player_display_name"].iloc[0] if len(name_row) else f"Player {pid}"
        out.append({
            "player_id": int(pid), "player_display_name": name,
            "appearances": int(len(arm)),
            "outs_recorded": int(arm["outs_recorded"].sum()),
            "strikeouts": int(arm["strikeouts"].sum()),
            "walks_allowed": int(arm["walks_allowed"].sum()),
            "hits_allowed": int(arm["hits_allowed"].sum()),
            "runs_allowed": int(arm["runs_allowed"].sum()),
            "avg_run_value": round(float(arm["run_value"].mean()), 3),
        })
    return out

"""Batter-vs-pitcher-handedness trailing feature: 'how has this batter hit
against RHP/LHP recently', matched to the handedness of whichever pitcher
they're actually facing next -- a more targeted signal than the blunt
team-wide opponent-allowed rate already used elsewhere. Walk-forward safe.
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "mlb"
WINDOW, MIN_GAMES = 20, 6  # PA-vs-one-hand samples are smaller, so a wider window


def pitcher_hand_lookup():
    """player_id -> dominant p_throws, from the pitch-profile table."""
    prof = pd.read_parquet(DATA_DIR / "pitcher_pitch_profile.parquet")
    counts = prof.groupby(["player_id", "p_throws"])["n_pitches"].sum().reset_index()
    idx = counts.groupby("player_id")["n_pitches"].idxmax()
    dominant = counts.loc[idx, ["player_id", "p_throws"]].rename(columns={"p_throws": "hand"})
    return dominant.set_index("player_id")["hand"]


def starters_by_team_date():
    """(team, game_date) -> starting pitcher's player_id."""
    pitchers = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    starters = pitchers[pitchers["is_starter"]].drop_duplicates(["team", "game_date"], keep="first")
    return starters[["team", "game_date", "player_id"]].rename(columns={"player_id": "starter_id"})


def attach_vs_hand_feature(df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """df must have columns: player_id, opponent_team, game_date. Adds
    own_vs_hand_trailing_avg: this player's trailing `stat_col` rate against
    the SAME handedness as the pitcher they're about to face (identified via
    the opposing team's starter for that game_date), as of strictly before
    game_date. Rows where we can't resolve a hand keep NaN (dropped by the
    caller, same as any other insufficient-history case)."""
    hand_lookup = pitcher_hand_lookup()
    starters = starters_by_team_date()

    df = df.merge(starters.rename(columns={"team": "opponent_team"}), on=["opponent_team", "game_date"], how="left")
    df["target_hand"] = df["starter_id"].map(hand_lookup)

    hand_logs = pd.read_parquet(DATA_DIR / "batter_vs_hand_logs.parquet")
    hand_logs = hand_logs.sort_values(["player_id", "vs_hand", "game_date"])
    hand_logs["trailing"] = hand_logs.groupby(["player_id", "vs_hand"])[stat_col].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
    )
    hand_trail = hand_logs[["player_id", "vs_hand", "game_date", "trailing"]].rename(
        columns={"vs_hand": "target_hand", "trailing": "own_vs_hand_trailing_avg"}
    ).sort_values("game_date")

    resolved = df.dropna(subset=["target_hand"]).sort_values("game_date")
    merged = pd.merge_asof(
        resolved, hand_trail, on="game_date", by=["player_id", "target_hand"], direction="backward",
    )
    unresolved = df[df["target_hand"].isna()].copy()
    unresolved["own_vs_hand_trailing_avg"] = pd.NA

    out = pd.concat([merged, unresolved], ignore_index=True).drop(columns=["starter_id", "target_hand"])
    return out

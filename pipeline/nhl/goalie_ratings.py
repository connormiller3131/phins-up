"""Trailing starting-goalie quality ratings, built from real per-game
boxscore data pulled by pull_goalie_games.py (api-web.nhle.com doesn't
expose a Statcast-style run-value metric the way MLB's Statcast feed does,
so this derives one: goals saved above what a league-average goalie would
have allowed on the same real shot volume that game -- positive = better
than average, same sign convention as MLB's pitcher run_value).

backtest_goals_model.py tested this (and several variants: shot-volume-
weighted, save%-rate, EWMA, across many window sizes) as a feature for
projected goals scored -- none of them beat the existing trailing-average
baseline on real held-out data (residual correlation never exceeded 0.03
in magnitude). So build_goalie_ratings/current_goalie_rating below are kept
only as the backtested record of that result; nothing in generate_current_
slate.py's projected_score consumes them. team_recent_save_pct is the part
that IS wired in, as a plain display-only Team Stats stat (not a model
input, no accuracy claim beyond "this is what really happened recently").
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nhl"

SG_WINDOW, SG_MIN = 6, 3  # trailing starts, same window MLB uses for starting pitchers


def _load_goalies():
    df = pd.read_parquet(DATA_DIR / "goalie_game_logs.parquet")
    return df[df["starter"]].dropna(subset=["shots_against", "saves"]).copy()


def _league_avg_save_pct(df):
    return float(df["saves"].sum() / df["shots_against"].sum())


def build_goalie_ratings():
    """One row per (player_id, team, game_date) for STARTING goalies:
    trailing average goals-saved-above-average per start, using only
    starts strictly before that date."""
    df = _load_goalies().sort_values(["player_id", "game_date"])
    league_avg = _league_avg_save_pct(df)
    df["gsaa"] = df["saves"] - df["shots_against"] * league_avg
    df["sg_rating"] = df.groupby("player_id")["gsaa"].transform(
        lambda s: s.shift(1).rolling(SG_WINDOW, min_periods=SG_MIN).mean()
    )
    return df[["player_id", "team", "game_date", "sg_rating"]], league_avg


def current_goalie_rating(player_id, league_avg=None):
    df = _load_goalies()
    if league_avg is None:
        league_avg = _league_avg_save_pct(df)
    starts = df[df["player_id"] == player_id].sort_values("game_date")
    if len(starts) < SG_MIN:
        return None
    recent = starts.tail(SG_WINDOW)
    gsaa = recent["saves"] - recent["shots_against"] * league_avg
    return float(gsaa.mean())


def team_recent_save_pct(team, window_games=10, min_games=3):
    """Real team-level goaltending save% over the last `window_games`
    distinct game dates (every goalie who appeared, not just starters --
    a display stat, not a model input, so there's no walk-forward-leakage
    concern the way there is for build_goalie_ratings). Returns None if the
    team has no logged appearances or fewer than min_games recent dates."""
    df = pd.read_parquet(DATA_DIR / "goalie_game_logs.parquet")
    team_df = df[df["team"] == team].dropna(subset=["shots_against", "saves"])
    if team_df.empty:
        return None
    recent_dates = team_df["game_date"].drop_duplicates().sort_values().tail(window_games)
    if len(recent_dates) < min_games:
        return None
    recent = team_df[team_df["game_date"].isin(recent_dates)]
    shots = recent["shots_against"].sum()
    if shots == 0:
        return None
    return float(recent["saves"].sum() / shots)


if __name__ == "__main__":
    ratings, league_avg = build_goalie_ratings()
    print(f"league average save%: {league_avg:.4f}")
    print(ratings.dropna().tail(10).to_string())

"""Feature engineering for MLB player-prop models: own trailing rate vs. the
upcoming opponent's trailing allowed-rate. Same walk-forward-safe design as
the NFL prop pipeline (shift(1) rolling window -- only games strictly before
the row's date are used)."""
import pathlib
import pandas as pd

from pipeline.mlb.player_names import get_name_lookup

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "mlb"
WINDOW = 15
MIN_GAMES = 5


def _trailing(series_by_group, window, min_games):
    return series_by_group.shift(1).rolling(window=window, min_periods=min_games).mean()


def _build(df, stat_col):
    names = get_name_lookup()
    df = df.merge(names, on="player_id", how="left")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    df["own_trailing_avg"] = df.groupby("player_id")[stat_col].transform(
        lambda s: _trailing(s, WINDOW, MIN_GAMES)
    )

    allowed = (
        df.groupby(["opponent_team", "game_date"])[stat_col]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team", stat_col: "allowed_that_game"})
        .sort_values(["defense_team", "game_date"])
    )
    allowed["opp_allowed_trailing_avg"] = allowed.groupby("defense_team")["allowed_that_game"].transform(
        lambda s: _trailing(s, WINDOW, MIN_GAMES)
    )

    df = df.merge(
        allowed[["defense_team", "game_date", "opp_allowed_trailing_avg"]],
        left_on=["opponent_team", "game_date"],
        right_on=["defense_team", "game_date"],
        how="left",
    )

    keep = [
        "player_id", "player_display_name", "team", "opponent_team", "game_date",
        stat_col, "own_trailing_avg", "opp_allowed_trailing_avg",
    ]
    out = df[keep].rename(columns={stat_col: "actual"})
    out = out.dropna(subset=["own_trailing_avg", "opp_allowed_trailing_avg", "player_display_name"])
    return out.reset_index(drop=True)


def build_batter_prop_table(stat_col: str):
    """stat_col in {hits, total_bases, home_runs, strikeouts}"""
    df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    return _build(df, stat_col)


def build_pitcher_prop_table():
    """Pitcher strikeouts."""
    df = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    return _build(df, "strikeouts")


if __name__ == "__main__":
    t = build_batter_prop_table("total_bases")
    print(t.shape)
    print(t.sort_values("game_date").tail(10))

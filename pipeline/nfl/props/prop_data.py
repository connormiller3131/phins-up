"""Feature engineering for player-prop models: own trailing rate vs. the
upcoming opponent's trailing allowed-rate to that position group. All trailing
stats use only games strictly before the row's game_date (walk-forward safe)."""
import pathlib
import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "nfl"
WINDOW = 8
MIN_GAMES = 3


def _load_base():
    ps = pl.read_parquet(DATA_DIR / "player_stats.parquet").to_pandas()
    sched = pl.read_parquet(DATA_DIR / "schedules.parquet").select(
        ["game_id", "gameday", "home_team", "away_team", "roof", "temp", "wind", "home_rest", "away_rest"]
    ).to_pandas()
    sched["game_date"] = pd.to_datetime(sched["gameday"])
    ps = ps.merge(sched, on="game_id", how="inner")
    ps["anytime_td"] = ((ps["rushing_tds"].fillna(0) + ps["receiving_tds"].fillna(0)) > 0).astype(float)

    ps["is_dome"] = ps["roof"].isin(["dome", "closed"]).astype(float)
    ps["own_rest"] = np.where(ps["team"] == ps["home_team"], ps["home_rest"], ps["away_rest"]).astype(float)

    # dome/closed games have no recorded weather (correctly -- there is none); use
    # controlled-environment values there, and the outdoor historical median elsewhere
    # (a fixed, historically-derived stand-in for "day-of forecast", since actual
    # future weather isn't knowable months ahead for live projections).
    outdoor = ps["is_dome"] == 0
    temp_fill = ps.loc[outdoor, "temp"].median()
    wind_fill = ps.loc[outdoor, "wind"].median()
    ps["temp"] = np.where(ps["is_dome"] == 1, 70.0, ps["temp"].fillna(temp_fill))
    ps["wind"] = np.where(ps["is_dome"] == 1, 0.0, ps["wind"].fillna(wind_fill))
    return ps


def _trailing(series_by_group, window, min_games):
    return series_by_group.shift(1).rolling(window=window, min_periods=min_games).mean()


def build_prop_table(stat_col: str, positions: list[str]):
    ps = _load_base()
    ps = ps[ps["position"].isin(positions)].copy()
    ps = ps.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # own trailing rate, per player, using only prior games
    ps["own_trailing_avg"] = ps.groupby("player_id")[stat_col].transform(
        lambda s: _trailing(s, WINDOW, MIN_GAMES)
    )

    # defense-allowed weekly totals to this position group, then trailing avg per defense
    allowed = (
        ps.groupby(["opponent_team", "season", "week", "game_date"])[stat_col]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team", stat_col: "allowed_that_week"})
        .sort_values(["defense_team", "game_date"])
    )
    allowed["opp_allowed_trailing_avg"] = allowed.groupby("defense_team")["allowed_that_week"].transform(
        lambda s: _trailing(s, WINDOW, MIN_GAMES)
    )

    ps = ps.merge(
        allowed[["defense_team", "season", "week", "opp_allowed_trailing_avg"]],
        left_on=["opponent_team", "season", "week"],
        right_on=["defense_team", "season", "week"],
        how="left",
    )

    keep = [
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "game_date", stat_col, "own_trailing_avg", "opp_allowed_trailing_avg",
        "is_dome", "temp", "wind", "own_rest",
    ]
    out = ps[keep].rename(columns={stat_col: "actual"})
    out = out.dropna(subset=["own_trailing_avg", "opp_allowed_trailing_avg"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    t = build_prop_table("receiving_yards", ["WR", "TE"])
    print(t.shape)
    print(t.sort_values("game_date").tail(10))

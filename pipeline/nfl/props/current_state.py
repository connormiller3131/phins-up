"""Current (as-of-today) trailing rates for players and defenses -- the
un-shifted version of the same rolling window used in prop_data.py, evaluated
at each player's/defense's most recent available game. Used to project a
future week that hasn't been played yet, where the model has to reason from
'their last N games' rather than a fixed backtest row."""
import pandas as pd

from pipeline.nfl.props.prop_data import _load_base, WINDOW, MIN_GAMES


def player_current_trailing(stat_col: str, positions: list[str]):
    ps = _load_base()
    ps = ps[ps["position"].isin(positions)].copy()
    ps = ps.sort_values(["player_id", "game_date"])

    ps["current_avg"] = ps.groupby("player_id")[stat_col].transform(
        lambda s: s.rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    ps["games_played"] = ps.groupby("player_id").cumcount() + 1

    latest = ps.sort_values("game_date").groupby("player_id").tail(1)
    return latest.set_index("player_id")[["player_display_name", "team", "current_avg", "games_played"]]


def defense_current_trailing(stat_col: str, positions: list[str]):
    ps = _load_base()
    ps = ps[ps["position"].isin(positions)].copy()

    allowed = (
        ps.groupby(["opponent_team", "season", "week", "game_date"])[stat_col]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team", stat_col: "allowed_that_week"})
        .sort_values(["defense_team", "game_date"])
    )
    allowed["current_avg"] = allowed.groupby("defense_team")["allowed_that_week"].transform(
        lambda s: s.rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    latest = allowed.sort_values("game_date").groupby("defense_team").tail(1)
    return latest.set_index("defense_team")["current_avg"]

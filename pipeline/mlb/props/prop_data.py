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


def _trailing_career(series_by_group, min_games):
    """Same walk-forward-safe shift(1) as _trailing, but an expanding (not
    capped-window) mean -- a player's full available history to date, not
    just their last WINDOW games. For a short-career player this is
    identical to (or barely diverges from) own_trailing_avg; for a veteran
    it captures whether a recent hot/cold stretch is really a change in
    form or just noise on top of a long, stable baseline rate."""
    return series_by_group.shift(1).expanding(min_periods=min_games).mean()


def _build(df, stat_col, volume_col=None, extra_trailing_cols=()):
    """volume_col: optional opportunity-volume column (pa_count for batters,
    batters_faced for pitchers) -> own_trailing_volume, the player's recent
    real playing-time level. A counting stat is rate x opportunities, and
    the original two features only ever saw the blended per-game rate --
    identical recent production from a 4.5-PA leadoff hitter and a lucky
    2-PA bench bat looked the same to the model.
    extra_trailing_cols: other per-game columns to expose as trailing means
    (e.g. est_woba), each as own_trailing_{col}. All use the same
    shift(1)/window/min_games as own_trailing_avg, so they're NaN on exactly
    the same rows and never change the training population."""
    names = get_name_lookup()
    df = df.merge(names, on="player_id", how="left")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    df["own_trailing_avg"] = df.groupby("player_id")[stat_col].transform(
        lambda s: _trailing(s, WINDOW, MIN_GAMES)
    )
    df["own_career_trailing_avg"] = df.groupby("player_id")[stat_col].transform(
        lambda s: _trailing_career(s, MIN_GAMES)
    )
    extra_cols = []
    if volume_col is not None:
        df["own_trailing_volume"] = df.groupby("player_id")[volume_col].transform(
            lambda s: _trailing(s, WINDOW, MIN_GAMES)
        )
        extra_cols.append("own_trailing_volume")
    for col in extra_trailing_cols:
        out_col = f"own_trailing_{col}"
        df[out_col] = df.groupby("player_id")[col].transform(
            lambda s: _trailing(s, WINDOW, MIN_GAMES)
        )
        extra_cols.append(out_col)

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
        stat_col, "own_trailing_avg", "own_career_trailing_avg", "opp_allowed_trailing_avg",
    ] + extra_cols
    out = df[keep].rename(columns={stat_col: "actual"})
    # own_career_trailing_avg uses the same min_games as own_trailing_avg, just
    # over an expanding instead of capped window, so it's never NaN when
    # own_trailing_avg isn't -- included here for safety, not because it
    # actually drops any additional rows. Same for the extra trailing
    # columns: identical shift/window/min_games on the same per-player
    # groupby means an identical NaN pattern, so requesting them never
    # changes which rows survive.
    out = out.dropna(subset=["own_trailing_avg", "own_career_trailing_avg", "opp_allowed_trailing_avg", "player_display_name"] + extra_cols)
    return out.reset_index(drop=True)


def build_batter_prop_table(stat_col: str, with_volume=False, extra_trailing_cols=()):
    """stat_col in {hits, total_bases, home_runs, strikeouts, walks, rbi}"""
    df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    return _build(df, stat_col, volume_col="pa_count" if with_volume else None,
                  extra_trailing_cols=extra_trailing_cols)


def build_pitcher_prop_table(stat_col: str = "strikeouts", with_volume=False):
    """stat_col in {strikeouts, hits_allowed, walks_allowed, runs_allowed, outs_recorded}"""
    df = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    return _build(df, stat_col, volume_col="batters_faced" if with_volume else None)


if __name__ == "__main__":
    t = build_batter_prop_table("total_bases")
    print(t.shape)
    print(t.sort_values("game_date").tail(10))

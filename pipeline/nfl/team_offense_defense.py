"""Trailing team-level passing/rushing offense and defense quality, built
from nflverse's real per-team-per-game boxscore stats (team_stats.parquet).
Same design as MLB's pitcher_ratings.py/team_offense.py: walk-forward safe
(shift(1) before the rolling window -- a rating never reflects the game it's
about to predict), with unshifted current_* versions for live/future
predictions.

Features, all yards-per-attempt or -per-carry (rate stats, not raw totals,
so a team's high volume alone doesn't masquerade as quality):
  - pass_ypa_off / rush_ypc_off: this team's own trailing efficiency
  - pass_ypa_def / rush_ypc_def: trailing efficiency ALLOWED by this team's
    defense (computed via a same-game self-join against the opponent's own
    row -- team_stats has no "yards allowed" column directly, just each
    team's own gained yards, so what team A allowed = what team B gained)
  - int_margin: this team's trailing (defensive INTs forced - offensive INTs
    thrown) per game, a net turnover-via-interception signal
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nfl"

WINDOW, MIN_GAMES = 8, 3


def _load_team_game_stats():
    """One row per (team, game_id) with own offensive rates and the
    opponent's same-game rates (== what this team's defense allowed),
    via a self-join on game_id. Includes both raw yards-per-play and
    EPA-per-play (Expected Points Added, which accounts for down/distance/
    situation rather than treating every yard the same) so the backtest can
    compare which is the more useful signal -- same "test the better metric,
    not just the obvious one" approach as MLB's est_woba vs. raw runs."""
    df = pd.read_parquet(DATA_DIR / "team_stats.parquet")
    df = df[["season", "week", "team", "game_id", "opponent_team",
              "attempts", "passing_yards", "passing_epa", "passing_interceptions",
              "carries", "rushing_yards", "rushing_epa", "def_interceptions"]].copy()

    df["pass_ypa_off"] = df["passing_yards"] / df["attempts"]
    df["rush_ypc_off"] = df["rushing_yards"] / df["carries"]
    df["pass_epa_pp_off"] = df["passing_epa"] / df["attempts"]
    df["rush_epa_pp_off"] = df["rushing_epa"] / df["carries"]

    opp = df[["game_id", "team", "pass_ypa_off", "rush_ypc_off", "pass_epa_pp_off", "rush_epa_pp_off"]].rename(
        columns={"team": "opponent_team", "pass_ypa_off": "pass_ypa_def", "rush_ypc_off": "rush_ypc_def",
                "pass_epa_pp_off": "pass_epa_pp_def", "rush_epa_pp_off": "rush_epa_pp_def"})
    merged = df.merge(opp, on=["game_id", "opponent_team"], how="inner")

    merged["int_margin_game"] = merged["def_interceptions"] - merged["passing_interceptions"]
    return merged


def build_offense_defense_ratings():
    """One row per (team, game_id): trailing (as-of-entering-that-game)
    offense/defense rates, using only games strictly before it."""
    df = _load_team_game_stats().sort_values(["team", "season", "week"])

    cols = [("pass_ypa_off", "pass_ypa_off_trail"), ("rush_ypc_off", "rush_ypc_off_trail"),
            ("pass_ypa_def", "pass_ypa_def_trail"), ("rush_ypc_def", "rush_ypc_def_trail"),
            ("pass_epa_pp_off", "pass_epa_pp_off_trail"), ("rush_epa_pp_off", "rush_epa_pp_off_trail"),
            ("pass_epa_pp_def", "pass_epa_pp_def_trail"), ("rush_epa_pp_def", "rush_epa_pp_def_trail"),
            ("int_margin_game", "int_margin_trail")]
    for col, out in cols:
        df[out] = df.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        )

    return df[["team", "season", "week", "game_id"] + [out for _, out in cols]]


def current_offense_defense_rating(team):
    """Unshifted trailing rates as of right now, for live/future predictions.
    Returns None for any rate the team doesn't have enough games for yet."""
    df = _load_team_game_stats()
    team_games = df[df["team"] == team].sort_values(["season", "week"])
    if len(team_games) < MIN_GAMES:
        return {"pass_ypa_off": None, "rush_ypc_off": None,
                "pass_ypa_def": None, "rush_ypc_def": None, "int_margin": None}

    recent = team_games.tail(WINDOW)
    return {
        "pass_ypa_off": float(recent["pass_ypa_off"].mean()),
        "rush_ypc_off": float(recent["rush_ypc_off"].mean()),
        "pass_ypa_def": float(recent["pass_ypa_def"].mean()),
        "rush_ypc_def": float(recent["rush_ypc_def"].mean()),
        "int_margin": float(recent["int_margin_game"].mean()),
    }


if __name__ == "__main__":
    ratings = build_offense_defense_ratings()
    print(ratings.shape)
    print(ratings.dropna().tail(10).to_string())

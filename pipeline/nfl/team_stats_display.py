"""Team Stats dropdown data: each team's own trailing offense (passing/
rushing/receiving) and defense (passing/rushing) splits, plus points for/
against -- a display feature, not a model input, so it's kept separate from
team_offense_defense.py (that module's rate features feed the win-prob
model and are validated by backtest; this one just surfaces real box-score
averages for the page, unshifted "as of right now" like generate_current_
week.py's current_team_scoring_rates).

Built from the same real nflverse team_stats.parquet team_offense_defense.py
already uses. "Yards/TDs allowed" aren't a column nflverse gives a team
directly -- they're derived via the same same-game self-join against the
opponent's own offensive row (what team B gained IS what team A's defense
allowed), already the established pattern here. Turnover-forcing stats
(def_interceptions, def_fumbles_forced) ARE already the team's own credited
defensive stat, no self-join needed for those.
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nfl"
WINDOW, MIN_GAMES = 8, 1  # display purposes: show a real number as soon as one game exists, not wait for 3 like the model features do


def _load_team_game_rows():
    df = pd.read_parquet(DATA_DIR / "team_stats.parquet")
    df = df[[
        "season", "week", "team", "game_id", "opponent_team",
        "attempts", "passing_yards", "passing_tds", "passing_interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "rushing_fumbles_lost", "sack_fumbles_lost", "receiving_fumbles_lost",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "def_interceptions", "def_fumbles_forced",
    ]].copy()
    df["fumbles_lost"] = (
        df["rushing_fumbles_lost"].fillna(0) + df["sack_fumbles_lost"].fillna(0) + df["receiving_fumbles_lost"].fillna(0)
    )

    # Defense-allowed columns: this team's opponent's own offensive row for
    # the same game_id, renamed to "*_allowed".
    opp = df[["game_id", "team", "attempts", "passing_yards", "passing_tds",
              "carries", "rushing_yards", "rushing_tds"]].rename(columns={
        "team": "opponent_team", "attempts": "att_faced", "passing_yards": "pass_yds_allowed",
        "passing_tds": "pass_td_allowed", "carries": "carries_faced",
        "rushing_yards": "rush_yds_allowed", "rushing_tds": "rush_td_allowed",
    })
    merged = df.merge(opp, on=["game_id", "opponent_team"], how="inner")
    return merged


def _load_points():
    """home_score/away_score per game_id, long-format by team -- reuses the
    same real schedules data generate_current_week.py's own scoring-rate
    function reads, just kept local here to avoid a circular import."""
    sched = pd.read_parquet(DATA_DIR / "schedules.parquet")
    home = sched[["game_id", "season", "week", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"})
    away = sched[["game_id", "season", "week", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"})
    return pd.concat([home, away], ignore_index=True)


def build_team_stats_table():
    """One row per (team, game_id): this team's trailing (unshifted, as-of-
    right-now) offense/defense splits and points for/against, ready to take
    the latest row per team for the upcoming slate."""
    df = _load_team_game_rows().sort_values(["team", "season", "week"])
    pts = _load_points()

    df = df.merge(pts[["game_id", "team", "points_for", "points_against"]], on=["game_id", "team"], how="left")

    rate_cols = {
        "passing_yards": "pass_yds", "attempts": "pass_att", "passing_tds": "pass_td", "passing_interceptions": "int_thrown",
        "rushing_yards": "rush_yds", "carries": "rush_att", "rushing_tds": "rush_td", "fumbles_lost": "fum_lost",
        "receiving_yards": "rec_yds", "receptions": "rec", "receiving_tds": "rec_td",
        "def_interceptions": "int_forced", "def_fumbles_forced": "fum_forced",
        "pass_yds_allowed": "pass_yds_allowed", "att_faced": "pass_att_faced", "pass_td_allowed": "pass_td_allowed",
        "rush_yds_allowed": "rush_yds_allowed", "carries_faced": "rush_att_faced", "rush_td_allowed": "rush_td_allowed",
        "points_for": "points_for", "points_against": "points_against",
    }
    for src, dst in rate_cols.items():
        df[f"{dst}_trail"] = df.groupby("team")[src].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        )
    return df


def current_team_stats(team_stats_table, team):
    """Latest trailing row for `team`, reshaped into the Offense/Defense/
    passing-rushing-receiving structure the frontend dropdown expects.
    Returns None if the team has no games yet this season (real gap, not
    guessed at)."""
    rows = team_stats_table[team_stats_table["team"] == team]
    if rows.empty:
        return None
    r = rows.iloc[-1]
    if pd.isna(r["pass_yds_trail"]):
        return None

    def rate(yds, att, decimals=1):
        y, a = r[yds], r[att]
        return round(float(y / a), decimals) if pd.notna(y) and pd.notna(a) and a > 0 else None

    return {
        "offense": {
            "passing": {
                "yds_per_game": round(float(r["pass_yds_trail"]), 1), "yds_per_att": rate("pass_yds_trail", "pass_att_trail"),
                "td_per_game": round(float(r["pass_td_trail"]), 2), "td_per_att": rate("pass_td_trail", "pass_att_trail", 3),
                "int_per_game": round(float(r["int_thrown_trail"]), 2),
            },
            "rushing": {
                "yds_per_game": round(float(r["rush_yds_trail"]), 1), "yds_per_att": rate("rush_yds_trail", "rush_att_trail"),
                "td_per_game": round(float(r["rush_td_trail"]), 2), "td_per_att": rate("rush_td_trail", "rush_att_trail", 3),
                "fum_per_game": round(float(r["fum_lost_trail"]), 2),
            },
            "receiving": {
                "yds_per_game": round(float(r["rec_yds_trail"]), 1), "yds_per_att": rate("rec_yds_trail", "rec_trail"),
                "td_per_game": round(float(r["rec_td_trail"]), 2), "td_per_att": rate("rec_td_trail", "rec_trail", 3),
            },
            "points_per_game": round(float(r["points_for_trail"]), 1) if pd.notna(r["points_for_trail"]) else None,
        },
        "defense": {
            "passing": {
                "yds_per_game": round(float(r["pass_yds_allowed_trail"]), 1) if pd.notna(r["pass_yds_allowed_trail"]) else None,
                "yds_per_att": rate("pass_yds_allowed_trail", "pass_att_faced_trail"),
                "td_per_game": round(float(r["pass_td_allowed_trail"]), 2) if pd.notna(r["pass_td_allowed_trail"]) else None,
                "td_per_att": rate("pass_td_allowed_trail", "pass_att_faced_trail", 3),
                "int_per_game": round(float(r["int_forced_trail"]), 2),
            },
            "rushing": {
                "yds_per_game": round(float(r["rush_yds_allowed_trail"]), 1) if pd.notna(r["rush_yds_allowed_trail"]) else None,
                "yds_per_att": rate("rush_yds_allowed_trail", "rush_att_faced_trail"),
                "td_per_game": round(float(r["rush_td_allowed_trail"]), 2) if pd.notna(r["rush_td_allowed_trail"]) else None,
                "td_per_att": rate("rush_td_allowed_trail", "rush_att_faced_trail", 3),
                "fum_per_game": round(float(r["fum_forced_trail"]), 2),
            },
            "points_per_game": round(float(r["points_against_trail"]), 1) if pd.notna(r["points_against_trail"]) else None,
        },
    }


if __name__ == "__main__":
    table = build_team_stats_table()
    print(table.shape)
    print(current_team_stats(table, "KC"))

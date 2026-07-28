"""Team Stats dropdown data for MLB: each team's trailing batting (offense)
and pitching (defense) per-game averages, adapted from NFL's passing/
rushing/receiving + interceptions/fumbles framing to baseball's own real
categories -- there's no batting equivalent of "yards per attempt" in the
data we have (Statcast's pitch-level feed gives at-bats-adjacent PA counts,
not true at-bats, so a real batting average isn't derivable here), so this
sticks to real per-game counting-stat averages rather than forcing a rate
stat that doesn't exist in the source data.

Aggregated from the same real per-player Statcast game logs already used
for player props (batter_game_logs.parquet / pitcher_game_logs.parquet),
summed to team-per-game totals, then trailing-averaged the same
shift(1)-then-rolling-mean way as every other trailing stat on this site.
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
WINDOW, MIN_GAMES = 10, 1  # display purposes: show a real number as soon as one game exists


def _team_batting_by_game():
    df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    return df.groupby(["team", "game_pk", "game_date"], as_index=False).agg(
        hits=("hits", "sum"), total_bases=("total_bases", "sum"), home_runs=("home_runs", "sum"),
        strikeouts=("strikeouts", "sum"), walks=("walks", "sum"), rbi=("rbi", "sum"),
    )


def _team_pitching_by_game():
    df = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    return df.groupby(["team", "game_pk", "game_date"], as_index=False).agg(
        strikeouts=("strikeouts", "sum"), hits_allowed=("hits_allowed", "sum"),
        walks_allowed=("walks_allowed", "sum"), runs_allowed=("runs_allowed", "sum"),
        outs_recorded=("outs_recorded", "sum"),
    )


def build_team_stats_table():
    """One row per (team, game_pk): trailing (unshifted, as-of-right-now)
    batting/pitching per-game averages, ready to take the latest row per
    team for the upcoming slate."""
    bat = _team_batting_by_game().sort_values(["team", "game_date"])
    pit = _team_pitching_by_game().sort_values(["team", "game_date"])

    for col in ["hits", "total_bases", "home_runs", "strikeouts", "walks", "rbi"]:
        bat[f"{col}_trail"] = bat.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
    for col in ["strikeouts", "hits_allowed", "walks_allowed", "runs_allowed", "outs_recorded"]:
        pit[f"{col}_trail"] = pit.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())

    return bat, pit


def current_team_stats(tables, team):
    """Latest trailing row for `team`, reshaped into an Offense (batting) /
    Defense (pitching) structure. Returns None if the team has no games yet
    (real gap, not guessed at)."""
    bat, pit = tables
    bat_rows = bat[bat["team"] == team]
    pit_rows = pit[pit["team"] == team]
    if bat_rows.empty or pit_rows.empty:
        return None
    b, p = bat_rows.iloc[-1], pit_rows.iloc[-1]
    if pd.isna(b["hits_trail"]) or pd.isna(p["strikeouts_trail"]):
        return None

    innings_trail = p["outs_recorded_trail"] / 3 if pd.notna(p["outs_recorded_trail"]) and p["outs_recorded_trail"] > 0 else None
    era = round(float(p["runs_allowed_trail"] * 9 / innings_trail), 2) if innings_trail else None

    return {
        "offense": {
            "batting": {
                "hits_per_game": round(float(b["hits_trail"]), 1),
                "total_bases_per_game": round(float(b["total_bases_trail"]), 1),
                "hr_per_game": round(float(b["home_runs_trail"]), 2),
                "bb_per_game": round(float(b["walks_trail"]), 1),
                "k_per_game": round(float(b["strikeouts_trail"]), 1),
                "rbi_per_game": round(float(b["rbi_trail"]), 1),
            },
        },
        "defense": {
            "pitching": {
                "era": era,
                "k_per_game": round(float(p["strikeouts_trail"]), 1),
                "hits_allowed_per_game": round(float(p["hits_allowed_trail"]), 1),
                "bb_allowed_per_game": round(float(p["walks_allowed_trail"]), 1),
                "runs_allowed_per_game": round(float(p["runs_allowed_trail"]), 1),
            },
        },
    }


if __name__ == "__main__":
    tables = build_team_stats_table()
    print(current_team_stats(tables, "NYY"))

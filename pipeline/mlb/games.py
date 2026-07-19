"""Build the core chronological MLB game table from the raw team schedule/
record scrape: one row per game (deduped from the 2x-per-game raw pull),
with real final scores and dates, plus per-team rest days and trailing
scoring form (both walk-forward safe -- shift(1) before any rolling/diff,
so a team's own current game never leaks into its own features)."""
import pathlib
import re
import numpy as np
import pandas as pd

from pipeline.mlb.team_map import br_to_statcast

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
FORM_WINDOW = 10
FORM_MIN_GAMES = 3


def _parse_date(date_str, season):
    # Baseball-Reference format: "Thursday, Mar 28" or "Friday, Sep 5 (1)" for doubleheaders
    cleaned = re.sub(r"\s*\(\d\)$", "", date_str)  # drop doubleheader game-number suffix
    cleaned = cleaned.split(",", 1)[1].strip()  # drop weekday name
    return pd.to_datetime(f"{cleaned} {season}", format="%b %d %Y", errors="coerce")


def _team_rest_and_form(games):
    """Long table, one row per team per game (each game contributes two rows:
    one from the home side, one from the away side), used to compute each
    team's own rest days and trailing scoring form off its own chronological
    schedule -- a team's away games count toward its rest/form just as much
    as its home games do."""
    home_side = pd.DataFrame({
        "season": games["season"], "game_date": games["game_date"],
        "team": games["home_team"], "runs_scored": games["home_score"], "runs_allowed": games["away_score"],
    })
    away_side = pd.DataFrame({
        "season": games["season"], "game_date": games["game_date"],
        "team": games["away_team"], "runs_scored": games["away_score"], "runs_allowed": games["home_score"],
    })
    long = pd.concat([home_side, away_side], ignore_index=True).sort_values(["team", "game_date"])

    long["prev_game_date"] = long.groupby("team")["game_date"].shift(1)
    long["rest_days"] = (long["game_date"] - long["prev_game_date"]).dt.days

    long["trailing_runs_scored"] = long.groupby("team")["runs_scored"].transform(
        lambda s: s.shift(1).rolling(window=FORM_WINDOW, min_periods=FORM_MIN_GAMES).mean()
    )
    long["trailing_runs_allowed"] = long.groupby("team")["runs_allowed"].transform(
        lambda s: s.shift(1).rolling(window=FORM_WINDOW, min_periods=FORM_MIN_GAMES).mean()
    )
    return long[["team", "game_date", "rest_days", "trailing_runs_scored", "trailing_runs_allowed"]]


def load_games():
    raw = pd.read_parquet(DATA_DIR / "team_schedule_raw.parquet")
    home = raw[raw["Home_Away"] != "@"].copy()
    home = home[home["R"].notna() & home["RA"].notna()]  # drop unplayed/future rows

    home["game_date"] = home.apply(lambda r: _parse_date(r["Date"], r["season"]), axis=1)
    home = home.dropna(subset=["game_date"])

    games = pd.DataFrame({
        "season": home["season"].astype(int),
        "game_date": home["game_date"],
        "home_team": home["team"].map(br_to_statcast),
        "away_team": home["Opp"].map(br_to_statcast),
        "home_score": home["R"].astype(int),
        "away_score": home["RA"].astype(int),
    })
    games["margin"] = games["home_score"] - games["away_score"]
    games["home_win"] = (games["margin"] > 0).astype(float)
    games = games.sort_values("game_date").reset_index(drop=True)

    # A team can play twice on the same calendar date (doubleheaders), so
    # rest/form has to be joined on (team, game_date, occurrence-within-day)
    # rather than a plain (team, game_date) merge, which would silently
    # duplicate rows for doubleheader games.
    games["_home_occ"] = games.groupby(["home_team", "game_date"]).cumcount()
    games["_away_occ"] = games.groupby(["away_team", "game_date"]).cumcount()

    form = _team_rest_and_form(games)
    form["_occ"] = form.groupby(["team", "game_date"]).cumcount()

    games = games.merge(
        form.rename(columns={"team": "home_team", "rest_days": "home_rest",
                             "trailing_runs_scored": "home_trailing_runs_scored",
                             "trailing_runs_allowed": "home_trailing_runs_allowed", "_occ": "_home_occ"}),
        on=["home_team", "game_date", "_home_occ"], how="left",
    )
    games = games.merge(
        form.rename(columns={"team": "away_team", "rest_days": "away_rest",
                             "trailing_runs_scored": "away_trailing_runs_scored",
                             "trailing_runs_allowed": "away_trailing_runs_allowed", "_occ": "_away_occ"}),
        on=["away_team", "game_date", "_away_occ"], how="left",
    )
    games = games.drop(columns=["_home_occ", "_away_occ"])
    return games


if __name__ == "__main__":
    df = load_games()
    print(df.shape)
    print(df["season"].value_counts().sort_index())
    print(df.tail(10))

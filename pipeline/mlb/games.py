"""Build the core chronological MLB game table from the raw team schedule/
record scrape: one row per game (deduped from the 2x-per-game raw pull),
with real final scores and dates."""
import pathlib
import re
import numpy as np
import pandas as pd

from pipeline.mlb.team_map import br_to_statcast

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"


def _parse_date(date_str, season):
    # Baseball-Reference format: "Thursday, Mar 28" or "Friday, Sep 5 (1)" for doubleheaders
    cleaned = re.sub(r"\s*\(\d\)$", "", date_str)  # drop doubleheader game-number suffix
    cleaned = cleaned.split(",", 1)[1].strip()  # drop weekday name
    return pd.to_datetime(f"{cleaned} {season}", format="%b %d %Y", errors="coerce")


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
    return games


if __name__ == "__main__":
    df = load_games()
    print(df.shape)
    print(df["season"].value_counts().sort_index())
    print(df.tail(10))

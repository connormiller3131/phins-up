"""Pull real historical NHL regular-season game results directly from
api-web.nhle.com's public (unauthenticated, no key needed) league-wide
schedule endpoint -- one JSON source for both schedule and results, unlike
MLB's separate Baseball-Reference (results) + Statcast (player data) split.

Steps by calendar week (the endpoint always returns a 7-day window) across
each season's regular-season date range, filters to gameType == 2 (regular
season only -- excludes preseason and playoffs) with a completed game state
("OFF" or "FINAL" both mean the game is over with a real final score, unlike
what the names suggest -- confirmed against real data, not assumed), and
dedupes by the league's own game id (each game only appears once in this
endpoint, unlike a per-team schedule pull which would show it twice).

Season windows are generous approximations (a few weeks of padding on each
side) rather than exact discovered boundaries -- the gameType filter does
the real work of excluding non-regular-season games, so slightly-off window
edges just mean a handful of extra empty weekly calls, not bad data. The
most recent season's window is the one exception: its exact real boundaries
(regularSeasonStartDate/EndDate) were confirmed directly against the API
rather than approximated, since a stale/wrong end date there would silently
drop real recent games that current team ratings actually depend on.
2020-21's delayed, shortened, realigned season and 2019-20's COVID-truncated
one are both included as real, if unusual, seasons -- same "keep the actual
weird season, don't special-case it away" convention as MLB's 2020.

Completed seasons never change once final, so each one is pulled exactly
once and cached to its own file forever after (same convention, and same
CI cache directory persistence, as MLB's pull_team_games.py) -- only the
current (most recent) season is re-pulled in full every run, since that's
the one that can still be mid-progress."""
import pathlib
import time
import requests
import pandas as pd

from pipeline.nhl.team_map import normalize_team

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nhl"
CACHE_DIR = DATA_DIR / "team_games_by_season"
BASE_URL = "https://api-web.nhle.com/v1/schedule"

# (season_start_year, window_start, window_end)
SEASON_WINDOWS = [
    (2018, "2018-10-01", "2019-04-12"),
    (2019, "2019-10-01", "2020-03-15"),  # COVID-truncated; no more regular-season games after the pause
    (2020, "2021-01-10", "2021-05-20"),  # delayed start, shortened 56-game realigned season
    (2021, "2021-10-01", "2022-05-01"),
    (2022, "2022-10-01", "2023-04-16"),
    (2023, "2023-10-01", "2024-04-20"),
    (2024, "2024-10-01", "2025-04-19"),
    (2025, "2025-10-01", "2026-04-20"),
]
CURRENT_SEASON = SEASON_WINDOWS[-1][0]


def _fetch_week(date_str):
    resp = requests.get(f"{BASE_URL}/{date_str}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def pull_season(season_start_year, window_start, window_end):
    games_by_id = {}
    d = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    while d <= end:
        data = _fetch_week(d.strftime("%Y-%m-%d"))
        for week in data.get("gameWeek", []):
            for g in week.get("games", []):
                if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                    continue
                away, home = g["awayTeam"], g["homeTeam"]
                if "score" not in away or "score" not in home:
                    continue
                games_by_id[g["id"]] = {
                    "season": season_start_year,
                    "game_date": week["date"],
                    "home_team": normalize_team(home["abbrev"]),
                    "away_team": normalize_team(away["abbrev"]),
                    "home_score": int(home["score"]),
                    "away_score": int(away["score"]),
                    "went_to_ot_so": g.get("periodDescriptor", {}).get("periodType") in ("OT", "SO"),
                }
        d += pd.Timedelta(days=7)
        time.sleep(0.2)  # polite pacing, no published rate limit but no reason to hammer it
    return list(games_by_id.values())


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_games = []
    for season_start_year, window_start, window_end in SEASON_WINDOWS:
        cache_file = CACHE_DIR / f"{season_start_year}.parquet"
        if season_start_year != CURRENT_SEASON and cache_file.exists():
            season_df = pd.read_parquet(cache_file)
            all_games.append(season_df)
            print(f"  {season_start_year}-{season_start_year+1}: reused cached pull "
                  f"({len(season_df)} games) -- season is final, never re-pulled")
            continue

        season_games = pull_season(season_start_year, window_start, window_end)
        season_df = pd.DataFrame(season_games)
        print(f"  {season_start_year}-{season_start_year+1}: {len(season_df)} regular-season games")
        if season_start_year != CURRENT_SEASON:
            season_df.to_parquet(cache_file)
        all_games.append(season_df)

    df = pd.concat(all_games, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "team_games.parquet"
    df.to_parquet(out_path)
    print(f"\nWrote {out_path} -- {len(df)} total games, seasons {sorted(df['season'].unique())}")
    print(df["season"].value_counts().sort_index())


if __name__ == "__main__":
    main()

"""Pull MLB team-level game results (schedule_and_record, Baseball-Reference)
for all 30 teams across several seasons, dedupe into one row per game, and
cache to parquet. Used for the team win-probability model."""
import pathlib
import time
import warnings
import pandas as pd
import pybaseball as pb

warnings.filterwarnings("ignore")
pb.cache.enable()

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Completed seasons never change once final, so each one is scraped exactly
# once and cached to its own file forever after (the cache dir is persisted
# across CI runs via actions/cache -- see .github/workflows/refresh.yml).
# Only the current season is live and gets re-pulled in full every run.
CACHE_DIR = DATA_DIR / "team_games_by_season"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]  # 2020 is the 60-game COVID season
CURRENT_SEASON = SEASONS[-1]

TEAMS_BY_SEASON = {
    season: [
        "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
        "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
        "ATH" if season >= 2025 else "OAK",
        "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
    ]
    for season in SEASONS
}


def main():
    rows = []
    for season in SEASONS:
        cache_file = CACHE_DIR / f"{season}.parquet"
        if season != CURRENT_SEASON and cache_file.exists():
            season_df = pd.read_parquet(cache_file)
            rows.append(season_df)
            print(f"season {season}: reused cached scrape ({len(season_df)} rows) -- season is final, never re-pulled")
            continue

        season_rows = []
        for team in TEAMS_BY_SEASON[season]:
            try:
                df = pb.schedule_and_record(season, team)
            except Exception as e:
                print(f"  skip {season} {team}: {e}")
                continue
            df = df.copy()
            df["season"] = season
            df["team"] = team
            season_rows.append(df)
            time.sleep(0.5)  # be polite to baseball-reference
        season_df = pd.concat(season_rows, ignore_index=True)
        if season != CURRENT_SEASON:
            # Only cache seasons that are actually over -- the current season
            # is re-pulled fresh every run regardless, so caching it would
            # just be a stale copy nothing ever reads.
            season_df.to_parquet(cache_file)
        rows.append(season_df)
        print(f"season {season}: pulled {len(TEAMS_BY_SEASON[season])} teams")

    all_games = pd.concat(rows, ignore_index=True)
    out_path = DATA_DIR / "team_schedule_raw.parquet"
    all_games.to_parquet(out_path)
    print(f"Wrote {len(all_games)} raw rows to {out_path}")


if __name__ == "__main__":
    main()

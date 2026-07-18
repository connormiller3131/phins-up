"""Pull Statcast pitch-level data month-by-month and aggregate immediately to
per-game batter and pitcher lines (hits, total bases, HRs, strikeouts) --
the MLB equivalent of nflreadpy's weekly player_stats table. Aggregating as
we go keeps peak memory bounded instead of holding 2-3 full seasons of raw
pitch data at once."""
import pathlib
import warnings
import calendar
import datetime
import numpy as np
import pandas as pd
import pybaseball as pb

warnings.filterwarnings("ignore")
pb.cache.enable()

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HIT_TB = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

TODAY = datetime.date.today()
SEASON_MONTHS = []
for year in (2024, 2025, 2026):
    for month in range(2, 12):  # Feb through Nov (spring training through World Series)
        start = datetime.date(year, month, 1)
        if start > TODAY:
            break
        end = datetime.date(year, month, calendar.monthrange(year, month)[1])
        end = min(end, TODAY)
        SEASON_MONTHS.append((start.isoformat(), end.isoformat()))


def aggregate_chunk(df):
    if df.empty:
        return None, None
    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return None, None

    pa["is_home_bat"] = pa["inning_topbot"] == "Bot"
    pa["batter_team"] = np.where(pa["is_home_bat"], pa["home_team"], pa["away_team"])
    pa["batter_opp"] = np.where(pa["is_home_bat"], pa["away_team"], pa["home_team"])
    pa["is_hit"] = pa["events"].isin(HIT_TB.keys()).astype(int)
    pa["tb"] = pa["events"].map(HIT_TB).fillna(0).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_k"] = pa["events"].str.contains("strikeout", na=False).astype(int)

    batter_game = (
        pa.groupby(["game_pk", "game_date", "batter", "batter_team", "batter_opp"])
        .agg(hits=("is_hit", "sum"), total_bases=("tb", "sum"), home_runs=("is_hr", "sum"),
             strikeouts=("is_k", "sum"), pa_count=("is_hit", "size"))
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "team", "batter_opp": "opponent_team"})
    )

    pitcher_game = (
        pa.groupby(["game_pk", "game_date", "pitcher", "batter_opp", "batter_team"])
        .agg(strikeouts=("is_k", "sum"), batters_faced=("is_k", "size"))
        .reset_index()
        .rename(columns={"pitcher": "player_id", "batter_opp": "team", "batter_team": "opponent_team"})
    )

    return batter_game, pitcher_game


def main():
    batter_chunks, pitcher_chunks = [], []
    for start, end in SEASON_MONTHS:
        print(f"Pulling {start} to {end}...")
        try:
            raw = pb.statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception as e:
            print(f"  skip: {e}")
            continue
        b, p = aggregate_chunk(raw)
        if b is not None:
            batter_chunks.append(b)
            pitcher_chunks.append(p)
            print(f"  -> {len(b)} batter-games, {len(p)} pitcher-games")
        del raw

    batters = pd.concat(batter_chunks, ignore_index=True)
    pitchers = pd.concat(pitcher_chunks, ignore_index=True)
    batters.to_parquet(DATA_DIR / "batter_game_logs.parquet")
    pitchers.to_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    print(f"\nWrote {len(batters)} batter-game rows and {len(pitchers)} pitcher-game rows")


if __name__ == "__main__":
    main()

"""Pull Statcast pitch-level data month-by-month and aggregate immediately to
three tables (keeping peak memory bounded instead of holding 2-3 full
seasons of raw pitch data at once):

1. batter_game_logs -- hits/TB/HR/K + est_woba (quality of contact, from
   Statcast's own expected-outcome model) per batter per game.
2. pitcher_game_logs -- strikeouts/batters_faced + run_value (sum of
   delta_pitcher_run_exp, Statcast's context-neutral run-expectancy
   contribution -- positive is good for the pitcher, confirmed empirically:
   a home run is about -1.7, a strikeout about +0.22) per pitcher per game,
   plus an is_starter flag (the pitcher with the most batters faced for
   that team that game -- an approximation that can misfire on true
   bullpen/opener games, a known simplification).
3. pitcher_pitch_profile -- monthly per-pitcher-per-pitch-type velocity/
   movement/usage, the raw ingredient for a pitcher "signature" used later
   to find batters' performance against similar pitchers.
"""
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


def aggregate_pitch_profile(df):
    if df.empty:
        return None
    pitch = df[df["pitch_type"].notna() & df["release_speed"].notna()].copy()
    if pitch.empty:
        return None
    pitch["month"] = pd.to_datetime(pitch["game_date"]).dt.to_period("M").astype(str)
    profile = (
        pitch.groupby(["pitcher", "month", "pitch_type", "p_throws"])
        .agg(n_pitches=("pitch_type", "size"),
             avg_speed=("release_speed", "mean"),
             avg_pfx_x=("pfx_x", "mean"),
             avg_pfx_z=("pfx_z", "mean"),
             avg_spin=("release_spin_rate", "mean"))
        .reset_index()
        .rename(columns={"pitcher": "player_id"})
    )
    return profile


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
             strikeouts=("is_k", "sum"), pa_count=("is_hit", "size"),
             est_woba=("estimated_woba_using_speedangle", "mean"))
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "team", "batter_opp": "opponent_team"})
    )

    pitcher_game = (
        pa.groupby(["game_pk", "game_date", "pitcher", "batter_opp", "batter_team"])
        .agg(strikeouts=("is_k", "sum"), batters_faced=("is_k", "size"),
             run_value=("delta_pitcher_run_exp", "sum"))
        .reset_index()
        .rename(columns={"pitcher": "player_id", "batter_opp": "team", "batter_team": "opponent_team"})
    )

    return batter_game, pitcher_game


def add_starter_flag(pitchers):
    idx = pitchers.groupby(["game_pk", "team"])["batters_faced"].idxmax()
    pitchers["is_starter"] = False
    pitchers.loc[idx, "is_starter"] = True
    return pitchers


def main():
    batter_chunks, pitcher_chunks, profile_chunks = [], [], []
    for start, end in SEASON_MONTHS:
        print(f"Pulling {start} to {end}...")
        try:
            raw = pb.statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception as e:
            print(f"  skip: {e}")
            continue

        profile = aggregate_pitch_profile(raw)
        if profile is not None:
            profile_chunks.append(profile)

        b, p = aggregate_chunk(raw)
        if b is not None:
            batter_chunks.append(b)
            pitcher_chunks.append(p)
            print(f"  -> {len(b)} batter-games, {len(p)} pitcher-games, "
                  f"{len(profile) if profile is not None else 0} pitch-profile rows")
        del raw

    batters = pd.concat(batter_chunks, ignore_index=True)
    pitchers = pd.concat(pitcher_chunks, ignore_index=True)
    pitchers = add_starter_flag(pitchers)
    profiles = pd.concat(profile_chunks, ignore_index=True)
    # collapse monthly chunks into one row per (pitcher, month, pitch_type, p_throws),
    # weighting the velocity/movement means by pitch count
    profiles["_w_speed"] = profiles["avg_speed"] * profiles["n_pitches"]
    profiles["_w_pfx_x"] = profiles["avg_pfx_x"] * profiles["n_pitches"]
    profiles["_w_pfx_z"] = profiles["avg_pfx_z"] * profiles["n_pitches"]
    profiles["_w_spin"] = profiles["avg_spin"] * profiles["n_pitches"]
    profiles = (
        profiles.groupby(["player_id", "month", "pitch_type", "p_throws"])
        .agg(n_pitches=("n_pitches", "sum"), _w_speed=("_w_speed", "sum"),
             _w_pfx_x=("_w_pfx_x", "sum"), _w_pfx_z=("_w_pfx_z", "sum"), _w_spin=("_w_spin", "sum"))
        .reset_index()
    )
    profiles["avg_speed"] = profiles["_w_speed"] / profiles["n_pitches"]
    profiles["avg_pfx_x"] = profiles["_w_pfx_x"] / profiles["n_pitches"]
    profiles["avg_pfx_z"] = profiles["_w_pfx_z"] / profiles["n_pitches"]
    profiles["avg_spin"] = profiles["_w_spin"] / profiles["n_pitches"]
    profiles = profiles.drop(columns=["_w_speed", "_w_pfx_x", "_w_pfx_z", "_w_spin"])

    batters.to_parquet(DATA_DIR / "batter_game_logs.parquet")
    pitchers.to_parquet(DATA_DIR / "pitcher_game_logs.parquet")
    profiles.to_parquet(DATA_DIR / "pitcher_pitch_profile.parquet")
    print(f"\nWrote {len(batters)} batter-game rows, {len(pitchers)} pitcher-game rows "
          f"({pitchers['is_starter'].sum()} flagged as starts), {len(profiles)} pitch-profile rows")


if __name__ == "__main__":
    main()

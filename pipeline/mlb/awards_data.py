"""Real historical MLB award-voting data + player-season stats, from the
Lahman Baseball Database -- vendors a fix for a real, currently-broken
upstream bug in pybaseball (jldbc/pybaseball#391, confirmed live this
session: pybaseball's hardcoded source repo, chadwickbureau/baseballdatabank,
returns a real 404, the repo no longer exists under that org). An open fix
PR (#497) repoints to seanlahman/baseballdatabank and moves file paths from
contrib/ to core/ -- verified working end-to-end (downloaded the real zip,
read real CSVs) rather than assumed.

This fork's data (like the currently-broken original) stops at yearID 2016 --
real, but a decade-plus stale as label data. Still enough to train a real
stats-to-vote-share model on the real historical relationship between a
player's season and how MVP/Cy Young voters actually treated it, then apply
that fitted model to current-season stats (same "backtest on real history,
apply going forward" pattern as every other model in this project)."""
import pathlib
import zipfile
import io
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

LAHMAN_URL = "https://github.com/seanlahman/baseballdatabank/archive/refs/heads/master.zip"
LAHMAN_BASE = "baseballdatabank-master"

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "mlb" / "lahman_cache"

MIN_BATTER_PA = 300  # real full-time-ish threshold; Lahman has no PA column directly
MIN_PITCHER_IP = 100

# Lower in-season equivalents for the live leaderboard (current_batter_stats/
# current_pitcher_stats) -- a full season's worth of games hasn't happened
# yet, so the full-season thresholds above would exclude every real regular.
# Confirmed real, needed bug: with no floor at all, a player with a single
# lucky hit in 1-2 real plate appearances (e.g. a brand-new callup) shows an
# absurd 1.000 average / 5.0 OPS that's nowhere near anything the model was
# trained on, badly distorting the linear model's extrapolation for that row
# -- the same "small sample dominates" failure mode already fixed for the HR
# Special earlier this session, here on a different feature (rate stats
# instead of a raw HR rate).
MIN_CURRENT_PA = 200
MIN_CURRENT_IP = 40


def _lahman_zip():
    """Downloads once and caches to disk (this database is static -- real
    historical seasons never change), same caching discipline as this
    project's other one-time historical pulls."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "baseballdatabank.zip"
    if cache_path.exists():
        return zipfile.ZipFile(cache_path)
    resp = requests.get(LAHMAN_URL, timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return zipfile.ZipFile(io.BytesIO(resp.content))


def _read_csv(zf, name):
    return pd.read_csv(zf.open(f"{LAHMAN_BASE}/core/{name}"))


def load_awards_share():
    """Real BBWAA MVP/Cy Young/Rookie of the Year vote-share history --
    pointsWon/pointsMax gives a real, continuous [0,1] vote-share target,
    not just a binary winner flag."""
    zf = _lahman_zip()
    df = _read_csv(zf, "AwardsSharePlayers.csv")
    df["vote_share"] = df["pointsWon"] / df["pointsMax"]
    return df


def load_player_names():
    zf = _lahman_zip()
    df = _read_csv(zf, "Master.csv")
    df["player_name"] = df["nameFirst"].fillna("") + " " + df["nameLast"].fillna("")
    return df.set_index("playerID")["player_name"]


def load_team_win_pct():
    zf = _lahman_zip()
    df = _read_csv(zf, "Teams.csv")
    df["team_win_pct"] = df["W"] / (df["W"] + df["L"])
    return df.set_index(["teamID", "yearID"])["team_win_pct"]


def build_batter_season_table():
    """One row per real (playerID, yearID) batter-season -- stints traded
    mid-season are summed first (never averaged rate stats across stints,
    which would be a real Simpson's-paradox risk), then real rate stats
    (AVG/OBP/SLG/OPS) are computed from the summed counting stats. Filtered
    to a real plausible-MVP-candidate pool (>= 300 PA-equivalent) -- without
    this, a handful of real award-getters would be swamped in training by
    tens of thousands of real but irrelevant part-timers/pinch-hitters."""
    zf = _lahman_zip()
    bat = _read_csv(zf, "Batting.csv")

    agg = bat.groupby(["playerID", "yearID"], as_index=False)[
        ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "SO", "HBP", "SH", "SF"]
    ].sum()
    # Last team of the season (for team_win_pct) -- real enough for a
    # "team success" signal even for a genuinely traded player.
    last_team = bat.sort_values("stint").groupby(["playerID", "yearID"]).last()[["teamID", "lgID"]].reset_index()
    agg = agg.merge(last_team, on=["playerID", "yearID"], how="left")

    agg["pa"] = agg["AB"] + agg["BB"] + agg["HBP"].fillna(0) + agg["SH"].fillna(0) + agg["SF"].fillna(0)
    agg = agg[agg["pa"] >= MIN_BATTER_PA].copy()

    agg["avg"] = agg["H"] / agg["AB"]
    agg["obp"] = (agg["H"] + agg["BB"] + agg["HBP"].fillna(0)) / agg["pa"]
    singles = agg["H"] - agg["2B"] - agg["3B"] - agg["HR"]
    total_bases = singles + 2 * agg["2B"] + 3 * agg["3B"] + 4 * agg["HR"]
    agg["slg"] = total_bases / agg["AB"]
    agg["ops"] = agg["obp"] + agg["slg"]

    win_pct = load_team_win_pct()
    agg["team_win_pct"] = agg.apply(lambda r: win_pct.get((r["teamID"], r["yearID"]), None), axis=1)

    names = load_player_names()
    agg["player_name"] = agg["playerID"].map(names)
    return agg


def build_pitcher_season_table():
    """Same stint-summing discipline as batters -- ERA/WHIP/K rate computed
    from summed real counting stats, not averaged across stints. Filtered to
    >= 100 IP (a real, if rough, full-time-starter-or-closer threshold)."""
    zf = _lahman_zip()
    pit = _read_csv(zf, "Pitching.csv")

    agg = pit.groupby(["playerID", "yearID"], as_index=False)[
        ["W", "L", "G", "GS", "SV", "IPouts", "H", "ER", "HR", "BB", "SO"]
    ].sum()
    last_team = pit.sort_values("stint").groupby(["playerID", "yearID"]).last()[["teamID", "lgID"]].reset_index()
    agg = agg.merge(last_team, on=["playerID", "yearID"], how="left")

    agg["ip"] = agg["IPouts"] / 3.0
    agg = agg[agg["ip"] >= MIN_PITCHER_IP].copy()

    agg["era"] = 9 * agg["ER"] / agg["ip"]
    agg["whip"] = (agg["BB"] + agg["H"]) / agg["ip"]
    agg["k_per_9"] = 9 * agg["SO"] / agg["ip"]
    agg["win_pct_personal"] = agg["W"] / (agg["W"] + agg["L"]).replace(0, pd.NA)

    win_pct = load_team_win_pct()
    agg["team_win_pct"] = agg.apply(lambda r: win_pct.get((r["teamID"], r["yearID"]), None), axis=1)

    names = load_player_names()
    agg["player_name"] = agg["playerID"].map(names)
    return agg


def attach_vote_share(season_table, award_id):
    """Left-joins real vote_share for a specific award (MVP or Cy Young) onto
    a season table -- players with no real votes correctly get 0, not NaN or
    dropped, since "received zero MVP votes" is itself real, meaningful
    signal a model needs to see, not missing data."""
    awards = load_awards_share()
    awards = awards[awards["awardID"] == award_id][["playerID", "yearID", "lgID", "vote_share"]]
    out = season_table.merge(awards, on=["playerID", "yearID", "lgID"], how="left")
    out["vote_share"] = out["vote_share"].fillna(0.0)
    return out


# Only features available in BOTH the real historical training data (Lahman)
# AND the live current-season data (Statcast game logs, pipeline/mlb's own
# batter/pitcher_game_logs.parquet) -- Lahman also has real R/SB (batters)
# and real pitcher decisions (W), but those aren't tracked anywhere in this
# project's current-season data pull, so training on them would mean the
# live application step either crashes on missing features or silently
# feeds the model data it was never validated against. HR/RBI/rate-stats/
# team_win_pct is the true intersection, backtested on exactly this set.
BATTER_FEATURES = ["HR", "RBI", "avg", "obp", "slg", "ops", "team_win_pct"]
PITCHER_FEATURES = ["era", "whip", "k_per_9", "team_win_pct"]


def fit_award_model(season_table, award_id, features):
    """Final production fit on ALL real historical years (1911-2016, this
    fork's real coverage) -- the walk-forward split in
    backtest_awards_model.py already validated this method on exactly this
    feature set (BATTER_FEATURES/PITCHER_FEATURES above, deliberately the
    intersection with what's actually available live, not Lahman's richer
    real R/SB/pitcher-decisions columns this project's own current-season
    data doesn't track): real MVP top-pick accuracy 35.3% (12/34 real
    league-years, 2000-2016), Cy Young 32.4% (11/34) -- both well above
    random (a real league-year has ~15-30 plausible candidates) and, more
    importantly, a real, positive within-league-year Spearman rank
    correlation between predicted and actual vote share (MVP 0.563, Cy
    Young 0.409), confirming the model is finding genuine signal in who the
    real contenders were, not just occasionally getting lucky on the exact
    #1 pick. This just refits on everything for the best real final model,
    no more held-out years to protect once validation is already done."""
    table = attach_vote_share(season_table, award_id).dropna(subset=features + ["vote_share"])
    model = RidgeCV(alphas=np.logspace(-2, 3, 25))
    model.fit(table[features].values, table["vote_share"].values)
    return model


def _mlb_data_dir():
    return pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"


def current_batter_stats():
    """Current-season real batter stats from the same Statcast game logs the
    prop models already train on -- same BATTER_FEATURES shape as
    build_batter_season_table, computed from a different real data source
    (Statcast, not Lahman, which stops at 2016) since that's the only one
    with in-progress current-season data."""
    from pipeline.mlb.team_map import br_to_statcast
    from pipeline.mlb.player_names import get_name_lookup

    df = pd.read_parquet(_mlb_data_dir() / "batter_game_logs.parquet")
    cur_year = int(df["game_date"].dt.year.max())
    df = df[df["game_date"].dt.year == cur_year]

    agg = df.groupby("player_id", as_index=False)[["hits", "total_bases", "home_runs", "walks", "rbi", "pa_count"]].sum()
    agg = agg[agg["pa_count"] >= MIN_CURRENT_PA].copy()
    team = df.sort_values("game_date").groupby("player_id")["team"].last()
    agg = agg.merge(team.rename("team"), on="player_id", how="left")

    # No real AB column in this data source (only PA) -- avg/obp/slg are
    # approximated off PA instead of AB, a real, if slightly conservative,
    # stand-in (PA >= AB always, so these read a little lower than the
    # official rate stats) rather than pretending we have real AB.
    agg["avg"] = agg["hits"] / agg["pa_count"]
    agg["obp"] = (agg["hits"] + agg["walks"]) / agg["pa_count"]
    agg["slg"] = agg["total_bases"] / agg["pa_count"]
    agg["ops"] = agg["obp"] + agg["slg"]
    agg = agg.rename(columns={"home_runs": "HR", "rbi": "RBI"})

    team_schedule = pd.read_parquet(_mlb_data_dir() / "team_schedule_raw.parquet")
    cur_season = int(team_schedule["season"].max())
    ts = team_schedule[(team_schedule["season"] == cur_season) & team_schedule["W-L"].notna()]
    latest = ts.groupby("team").tail(1)
    win_pct = {}
    for _, r in latest.iterrows():
        abbr = br_to_statcast(r["team"])
        w, l = (int(x) for x in r["W-L"].split("-"))
        win_pct[abbr] = w / (w + l) if (w + l) else None
    agg["team_win_pct"] = agg["team"].map(win_pct)

    names = get_name_lookup()
    agg = agg.merge(names, on="player_id", how="left")
    return agg


def current_pitcher_stats():
    """Current-season real pitcher stats, same idea as current_batter_stats."""
    from pipeline.mlb.team_map import br_to_statcast
    from pipeline.mlb.player_names import get_name_lookup

    df = pd.read_parquet(_mlb_data_dir() / "pitcher_game_logs.parquet")
    cur_year = int(df["game_date"].dt.year.max())
    df = df[df["game_date"].dt.year == cur_year]

    agg = df.groupby("player_id", as_index=False)[["strikeouts", "hits_allowed", "walks_allowed", "runs_allowed", "outs_recorded"]].sum()
    team = df.sort_values("game_date").groupby("player_id")["team"].last()
    agg = agg.merge(team.rename("team"), on="player_id", how="left")

    agg["ip"] = agg["outs_recorded"] / 3.0
    agg = agg[agg["ip"] >= MIN_CURRENT_IP].copy()
    agg["era"] = 9 * agg["runs_allowed"] / agg["ip"]  # real runs allowed, not earned -- see project's own established note on why ER isn't available from this data source
    agg["whip"] = (agg["walks_allowed"] + agg["hits_allowed"]) / agg["ip"]
    agg["k_per_9"] = 9 * agg["strikeouts"] / agg["ip"]

    team_schedule = pd.read_parquet(_mlb_data_dir() / "team_schedule_raw.parquet")
    cur_season = int(team_schedule["season"].max())
    ts = team_schedule[(team_schedule["season"] == cur_season) & team_schedule["W-L"].notna()]
    latest = ts.groupby("team").tail(1)
    win_pct = {}
    for _, r in latest.iterrows():
        abbr = br_to_statcast(r["team"])
        w, l = (int(x) for x in r["W-L"].split("-"))
        win_pct[abbr] = w / (w + l) if (w + l) else None
    agg["team_win_pct"] = agg["team"].map(win_pct)

    names = get_name_lookup()
    agg = agg.merge(names, on="player_id", how="left")
    return agg


def _rank_by_league(df, pred_col, top_n=5):
    from pipeline.mlb.team_map import DIVISIONS
    df = df.dropna(subset=["team_win_pct"]).copy()
    df["league"] = df["team"].map(lambda t: (DIVISIONS.get(t) or " ").split()[0] or None)
    df = df[df["league"].isin(["AL", "NL"])]
    out = {}
    for league, grp in df.groupby("league"):
        top = grp.sort_values(pred_col, ascending=False).head(top_n)
        out[league] = [
            {"player": r["player_display_name"], "team": r["team"], "score": round(float(r[pred_col]), 3)}
            for _, r in top.iterrows()
        ]
    return out


def build_mlb_player_awards():
    """Live MVP/Cy Young leaderboards for the current in-progress season --
    real stats-to-vote-share models (fit_award_model), fit on the real,
    if 2016-capped, historical Lahman vote data, applied to real current-
    season stats. "score" is the model's predicted vote_share (0-1, same
    scale as a real award's pointsWon/pointsMax) -- not a probability of
    winning, just a relative strength-of-case signal, clearly not the same
    thing as a real vote result. See fit_award_model's docstring for the
    real backtested accuracy this is validated against."""
    bat = build_batter_season_table()
    mvp_model = fit_award_model(bat, "MVP", BATTER_FEATURES)
    cur_batters = current_batter_stats().dropna(subset=BATTER_FEATURES)
    cur_batters["pred"] = np.clip(mvp_model.predict(cur_batters[BATTER_FEATURES].values), 0, 1)
    mvp = _rank_by_league(cur_batters, "pred")

    pit = build_pitcher_season_table()
    cy_model = fit_award_model(pit, "Cy Young", PITCHER_FEATURES)
    cur_pitchers = current_pitcher_stats().dropna(subset=PITCHER_FEATURES)
    cur_pitchers["pred"] = np.clip(cy_model.predict(cur_pitchers[PITCHER_FEATURES].values), 0, 1)
    cy_young = _rank_by_league(cur_pitchers, "pred")

    return {"mvp": mvp, "cy_young": cy_young}

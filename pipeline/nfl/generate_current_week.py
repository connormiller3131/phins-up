"""Generate REAL projections for the NFL season, all weeks:
- Elo win probability, ratings carried forward from all completed games
  through the fitted hyperparameters (no re-fitting on future weeks).
- Real posted opening market lines (moneyline/spread/total) where already
  posted in nflverse schedules -- null for weeks too far out for books to
  have priced yet, same null-handling pattern used everywhere else.
- Player props for each team's current depth-chart starters (QB/RB1/WR1),
  using their actual current trailing rate vs. the opponent defense's actual
  current trailing allowed-rate, fit ONCE on the full historical dataset and
  reused across every week (every target week is in the future relative to
  that fit, so there's no leakage to guard against the way there is in a
  backtest).
- Game metadata: stadium, location, weekday/kickoff time, and a primetime
  (TNF/SNF/MNF) flag, all already present in nflverse schedules.
"""
import sys
import pathlib
import json
import datetime
import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.games import load_games, moneyline_to_prob
from pipeline.nfl.elo_model import run_elo
from pipeline.nfl.props.prop_data import build_prop_table, WINDOW
from pipeline.nfl.props.current_state import player_current_trailing, defense_current_trailing
from pipeline.nfl.props.prop_models import (
    FEATURES, PROP_CONFIG, prop_features, prop_over_prob, yardage_over_prob,
)
from pipeline.common.count_dist import estimate_dispersion
from pipeline.nfl.props.nfl_td_odds import fetch_current_week_odds_map, attach_current_lines, attach_td_odds
from pipeline.nfl.team_stats_display import build_team_stats_table, current_team_stats
from pipeline.common.odds_history import record_title_odds
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

DATA_DIR = ROOT / "data" / "nfl"
RESULTS_DIR = ROOT / "docs" / "results"

# Identifies the over/under machinery behind model_over_prob, stamped into
# every frozen prediction snapshot. "mixed-v1" is the per-stat split
# validated in pipeline/nfl/props/backtest_count_volume.py: negative
# binomial for the small discrete counts, empirical residuals for the
# right-skewed yardage props, Normal where it still wins. Bump this whenever
# that machinery changes, so graded snapshots from different models are
# never pooled into one accuracy or calibration number.
PROP_PROB_MODEL = "mixed-v1"


def detect_target_week():
    """Find the next not-yet-played REG-season week: earliest (season, week)
    with a null home_score and gameday >= today. Avoids relying on
    nflreadpy's get_current_season/get_current_week, which track the most
    recently completed data rather than what's coming up next."""
    import nflreadpy as nfl

    today = datetime.date.today()
    candidate_seasons = [today.year - 1, today.year, today.year + 1]
    sched = nfl.load_schedules(seasons=candidate_seasons).to_pandas()
    sched = sched[sched["game_type"] == "REG"]
    sched["game_date"] = pd.to_datetime(sched["gameday"]).dt.date

    upcoming = sched[(sched["home_score"].isna()) & (sched["game_date"] >= today)]
    if upcoming.empty:
        raise RuntimeError("No upcoming unplayed REG-season games found in the schedule feed.")

    upcoming = upcoming.sort_values("game_date")
    return int(upcoming.iloc[0]["season"]), int(upcoming.iloc[0]["week"])


def team_names():
    import nflreadpy as nfl
    t = nfl.load_teams().to_pandas()
    return dict(zip(t["team_abbr"], t["team_name"]))


def get_season_schedule(target_season):
    import nflreadpy as nfl
    sched = nfl.load_schedules(seasons=[target_season]).to_pandas()
    sched = sched[sched["game_type"] == "REG"].copy()
    sched["market_home_prob_raw_away"] = moneyline_to_prob(sched["away_moneyline"].values)
    sched["market_home_prob_raw_home"] = moneyline_to_prob(sched["home_moneyline"].values)
    overround = sched["market_home_prob_raw_away"] + sched["market_home_prob_raw_home"]
    sched["market_home_prob"] = sched["market_home_prob_raw_home"] / overround
    return sched


def primetime_label(weekday, gametime):
    """TNF/SNF/MNF badge from the day of week + kickoff time. gametime is a
    24h 'HH:MM' local-to-stadium string in nflverse schedules."""
    if not weekday or not gametime:
        return None
    try:
        hour = int(str(gametime).split(":")[0])
    except (ValueError, IndexError):
        hour = None
    if weekday == "Thursday":
        return "TNF"
    if weekday == "Monday":
        return "MNF"
    if weekday == "Sunday" and hour is not None and hour >= 18:
        return "SNF"
    return None






def elo_predictions_for_season(games_df, season_sched):
    """Run Elo once across completed history + every future game in the
    season schedule (chronologically appended), returning predictions
    aligned to season_sched's row order."""
    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    future_rows = pd.DataFrame({
        "season": season_sched["season"].values,
        "week": season_sched["week"].values,
        "home_team": season_sched["home_team"].values,
        "away_team": season_sched["away_team"].values,
        "margin": np.nan,
        "home_win": np.nan,
        "location": season_sched["location"].values,
        "home_rest": season_sched["home_rest"].values,
        "away_rest": season_sched["away_rest"].values,
    })
    # only rows not already completed belong in the "future" tail; completed
    # games from this same season (shouldn't normally happen pre-kickoff,
    # but keep it correct if this runs mid-season) are already in games_df.
    already_played_keys = set(zip(games_df["season"], games_df["week"], games_df["home_team"], games_df["away_team"]))
    future_rows = future_rows[~future_rows.apply(
        lambda r: (r["season"], r["week"], r["home_team"], r["away_team"]) in already_played_keys, axis=1)]

    cols = ["season", "week", "home_team", "away_team", "margin", "home_win", "location", "home_rest", "away_rest"]
    combined = pd.concat([games_df[cols], future_rows], ignore_index=True)
    preds = run_elo(combined, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"],
                    rest_adv=elo_params.get("rest_adv", 0.0), season_regression=elo_params.get("season_regression", 0.75))

    n_future = len(future_rows)
    future_preds = dict(zip(
        zip(future_rows["season"], future_rows["week"], future_rows["home_team"], future_rows["away_team"]),
        preds[-n_future:] if n_future else [],
    ))
    # completed games in this season already have a real outcome-based prob
    # from the main Elo run; for a pre-season page these won't exist yet.
    played_mask = games_df["season"] == season_sched["season"].iloc[0]
    played_preds = {}
    if played_mask.any():
        played_idx = np.where(played_mask.values)[0]
        for idx in played_idx:
            row = games_df.iloc[idx]
            played_preds[(row["season"], row["week"], row["home_team"], row["away_team"])] = preds[idx]

    out = []
    for row in season_sched.itertuples(index=False):
        key = (row.season, row.week, row.home_team, row.away_team)
        out.append(future_preds.get(key, played_preds.get(key)))
    return out, elo_params


STARTER_DEPTH = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}  # how many ranks deep to pull per position


def get_starters(target_season):
    """Full starting-offense depth chart per team: QB1, RB1-2, WR1-3, TE1."""
    import nflreadpy as nfl
    dc = nfl.load_depth_charts(seasons=[target_season]).to_pandas()
    latest_dt = dc["dt"].max()
    dc = dc[dc["dt"] == latest_dt]

    starters = {}  # team -> {"QB": [gsis_id], "RB": [gsis_id, ...], "WR": [...], "TE": [...]}
    for team, grp in dc.groupby("team"):
        picks = {}
        for pos, depth in STARTER_DEPTH.items():
            rows = grp[(grp["pos_abb"] == pos) & (grp["pos_rank"] <= depth)].sort_values("pos_rank")
            ids = [r.gsis_id for r in rows.itertuples(index=False) if pd.notna(r.gsis_id)]
            if ids:
                picks[pos] = ids
        starters[team] = picks
    return starters, latest_dt


YARDAGE_LADDER_OFFSETS = (-20, -10, 0, 10, 20)


def yardage_ladder(prep, pred_mean):
    """Lines in steps of 10 around the model's own predicted mean, e.g.
    190/200/210/220/230 for a game the model projects at ~207 yds. Priced
    with the same distribution as the main line (prop_over_prob), so a
    ladder rung and the headline number can never disagree about shape."""
    base = max(round(pred_mean / 10) * 10, 10)
    ladder = []
    for off in YARDAGE_LADDER_OFFSETS:
        line = base + off
        if line <= 0:
            continue
        ladder.append({"line": float(line), "over_prob": round(prop_over_prob(prep, pred_mean, line), 3)})
    return ladder


def prepare_count_model(stat_col, positions):
    """Fit a RidgeCV model ONCE on full history for any continuous/count stat
    (yards, completions, attempts, carries, receptions...). Doesn't depend on
    the target week, so every game/week across the whole season reuses it.

    Also carries whatever this stat's backtested over/under distribution
    needs (see prop_models.PROP_CONFIG): a pooled residual std for Normal
    stats, a negative-binomial dispersion for the small discrete counts, or
    the full sorted residual vector for the empirical yardage props."""
    cfg = PROP_CONFIG[stat_col]
    features = prop_features(stat_col)
    hist = build_prop_table(stat_col, positions, volume_col=cfg["volume"])
    model = RidgeCV(alphas=np.logspace(-1, 3, 25))
    X, y = hist[features].values, hist["actual"].values
    model.fit(X, y)
    fitted = model.predict(X)
    resid = y - fitted
    return {
        "model": model,
        "features": features,
        "dist": cfg["dist"],
        "volume": cfg["volume"],
        "resid_std": max(float(np.std(resid)), 1e-6),
        "dispersion": float(estimate_dispersion(y, fitted)),
        "resid_sorted": np.sort(resid),
        "own": player_current_trailing(stat_col, positions, volume_col=cfg["volume"]),
        "defense": defense_current_trailing(stat_col, positions),
    }


def project_count(prep, player_id, opp_team, env, with_ladder=False):
    own, defense = prep["own"], prep["defense"]
    if player_id not in own.index or opp_team not in defense.index:
        return None

    own_avg = float(own.loc[player_id, "current_avg"])
    opp_avg = float(defense.loc[opp_team])
    if pd.isna(own_avg) or pd.isna(opp_avg):
        return None  # fewer than MIN_GAMES of trailing history (rookie/deep backup) -- no basis to project
    row = [own_avg, opp_avg, env["is_dome"], env["temp"], env["wind"], env["own_rest"], env["implied_team_total"]]
    if prep.get("volume"):
        own_vol = float(own.loc[player_id, "current_volume"])
        if pd.isna(own_vol):
            return None  # same reason as above, for the opportunity feature
        row.append(own_vol)
    pred_mean = float(prep["model"].predict([row])[0])
    # Anchored on the model's own predicted mean (which already blends own
    # average, opponent, weather, rest, and implied team total), not the
    # player's raw own-average alone -- same fix as MLB's project_count_stat.
    # Anchoring on own_avg let a below-average player's line round down to
    # an easy bar relative to what the model actually expected (e.g. a
    # plus matchup or a dome game the raw average doesn't capture), while a
    # real workhorse's higher own_avg rounded up to a harder one -- same
    # "probability of clearing your own line ends up negatively correlated
    # with real production" issue confirmed on MLB's real data.
    line = round(pred_mean * 2) / 2
    over_prob = prop_over_prob(prep, pred_mean, line)
    out = {
        "line": line, "projected": round(pred_mean, 1), "model_over_prob": round(over_prob, 3),
        "player_display_name": own.loc[player_id, "player_display_name"],
        "games_played": int(own.loc[player_id, "games_played"]),
    }
    if with_ladder:
        out["ladder"] = yardage_ladder(prep, pred_mean)
    return out


# The anytime-TD model trains on the shared seven PLUS an opportunity
# feature (own_trailing_volume, from prop_data's carries+targets
# "opportunities" column). Added because the seven-feature version had no
# way to tell a low-usage player on a scoring heater from a genuine every-
# down threat -- both present as the same trailing TD rate -- which is
# exactly the rate-vs-opportunities split PROP_CONFIG already handles for
# every count prop. Justified by walk-forward backtest in
# pipeline/nfl/props/backtest_td_volume.py, not by assumption: on the
# untouched 2025 season, Brier 0.1448 -> 0.1409 and log loss 0.4567 ->
# 0.4443. The whole-population move is small because anytime TD is a ~19%
# base-rate market; the tails are where it matters. The old model
# understated genuine workhorses (8+ carries+targets per game: predicted
# 0.300, actually scored 0.377) as badly as it overstated depth pieces
# (0-4 opp/gm: predicted 0.145, actually 0.082), and the opportunity
# feature closes the workhorse gap almost entirely (-0.077 -> -0.008)
# while cutting the depth-piece one by roughly a third (+0.063 -> +0.037).
# Replaying the TD SPECIAL's own 3-legs-3-teams rule week by week over
# that season, its legs went from 36/65 to 40/65.
#
# It does NOT fully fix the low-usage tail (+0.037 is still overstated),
# which is why dashboard_live.html ALSO gates who may be featured on the
# card rather than trusting the probability alone.
TD_FEATURES = FEATURES + ["own_trailing_volume"]
TD_VOLUME_COL = "opportunities"


def prepare_td_model():
    hist = build_prop_table("anytime_td", ["RB", "WR", "TE"], volume_col=TD_VOLUME_COL)
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(hist[TD_FEATURES].values, hist["actual"].values)
    return {
        "model": model,
        "own": player_current_trailing("anytime_td", ["RB", "WR", "TE"], volume_col=TD_VOLUME_COL),
        "defense": defense_current_trailing("anytime_td", ["RB", "WR", "TE"]),
    }


def project_td(prep, player_id, opp_team, env):
    own, defense = prep["own"], prep["defense"]
    if player_id not in own.index or opp_team not in defense.index:
        return None

    own_avg = float(own.loc[player_id, "current_avg"])
    opp_avg = float(defense.loc[opp_team])
    opp_per_gm = float(own.loc[player_id, "current_volume"])
    if pd.isna(own_avg) or pd.isna(opp_avg) or pd.isna(opp_per_gm):
        return None
    feat_row = [[own_avg, opp_avg, env["is_dome"], env["temp"], env["wind"], env["own_rest"],
                 env["implied_team_total"], opp_per_gm]]
    prob = float(prep["model"].predict_proba(feat_row)[:, 1][0])
    # trailing_n is the size of the window that actually produced own_avg,
    # not career games -- capped at WINDOW because that is all the rolling
    # mean ever looks at. Named to match the MLB props' own field so the
    # frontend's existing isReliablePick works on NFL TD legs unchanged.
    games_played = int(own.loc[player_id, "games_played"])
    return {"model_prob": round(prob, 3), "games_played": games_played,
            "trailing_n": min(games_played, WINDOW), "opp_per_gm": round(opp_per_gm, 1)}


def _prop_entry(section, market, team, opp_team, player_id, r, ladder=False):
    e = {"section": section, "player": r["player_display_name"], "player_id": player_id, "team": team,
         "opp": opp_team, "market": market, "line": r["line"], "projected": r["projected"],
         "model_over_prob": r["model_over_prob"]}
    if ladder and r.get("ladder"):
        e["ladder"] = r["ladder"]
    return e


def _td_entry(section, team, opp_team, player_id, player_name, t):
    # trailing_n and opp_per_gm are carried through so the TD SPECIAL card
    # can tell a featured pick apart from a thin-sample or low-usage one.
    # project_td always computed the sample size; it used to be dropped here,
    # which left the frontend's reliability gate inert for every NFL TD leg.
    return {"section": section, "player": player_name, "player_id": player_id, "team": team,
            "opp": opp_team, "market": "Anytime TD", "model_prob": t["model_prob"],
            "trailing_n": t["trailing_n"], "opp_per_gm": t["opp_per_gm"]}


# Report statuses that mean "will not play". Questionable is deliberately NOT
# here: the large majority of players listed Questionable do play, so treating
# it as an absence would remove far more real projections than it saves. It
# still gets surfaced as a badge so the reader can apply their own judgment.
INJURY_EXCLUDE = {"Out"}
INJURY_BADGE = {"Out", "Doubtful", "Questionable"}


def load_injury_status(season):
    """{(week, gsis_id): report_status} for one season, or {} when there is
    no report yet.

    Injury reports only exist for weeks at or near the present, while this
    generator projects all 18 weeks. That asymmetry is handled by keying on
    week: a future week simply has no entries and is left unfiltered, rather
    than borrowing some other week's report and implying knowledge the data
    does not have. A whole season with no reports (a season that has not
    started, which is the case for 2026 as of this writing) is the same
    no-op at larger scale."""
    path = DATA_DIR / "injuries.parquet"
    if not path.exists():
        print("  no injuries.parquet -- prop injury filtering skipped")
        return {}
    inj = pl.read_parquet(path).to_pandas()
    inj = inj[(inj["season"] == season) & inj["report_status"].notna() & inj["gsis_id"].notna()]
    if inj.empty:
        print(f"  no {season} injury reports yet -- prop injury filtering is a no-op for now")
        return {}
    out = {(int(r.week), r.gsis_id): r.report_status
           for r in inj.itertuples(index=False)}
    n_out = sum(1 for v in out.values() if v in INJURY_EXCLUDE)
    print(f"  injury reports: {len(out)} player-weeks for {season} ({n_out} ruled Out)")
    return out


def build_props_for_team(team, opp_team, starters, env, models, injuries=None, week=None):
    entries = []
    picks = starters.get(team, {})
    injuries = injuries or {}

    def status_for(pid):
        return injuries.get((week, pid)) if week is not None else None

    def is_out(pid):
        return status_for(pid) in INJURY_EXCLUDE

    for qb_id in picks.get("QB", []):
        if is_out(qb_id):
            continue
        r = project_count(models["passing_yards"], qb_id, opp_team, env, with_ladder=True)
        if r:
            entries.append(_prop_entry("Passing", "Passing Yds", team, opp_team, qb_id, r, ladder=True))
        rt = project_count(models["passing_tds"], qb_id, opp_team, env)
        if rt:
            entries.append(_prop_entry("Passing", "Passing TDs", team, opp_team, qb_id, rt))
        rc = project_count(models["completions"], qb_id, opp_team, env)
        if rc:
            entries.append(_prop_entry("Passing", "Completions", team, opp_team, qb_id, rc))
        ra = project_count(models["attempts"], qb_id, opp_team, env)
        if ra:
            entries.append(_prop_entry("Passing", "Pass Attempts", team, opp_team, qb_id, ra))

    for rb_id in picks.get("RB", []):
        if is_out(rb_id):
            continue
        r = project_count(models["rushing_yards"], rb_id, opp_team, env, with_ladder=True)
        if r:
            entries.append(_prop_entry("Rushing", "Rushing Yds", team, opp_team, rb_id, r, ladder=True))
        rc = project_count(models["carries"], rb_id, opp_team, env)
        if rc:
            entries.append(_prop_entry("Rushing", "Carries", team, opp_team, rb_id, rc))
        t = project_td(models["td"], rb_id, opp_team, env)
        if t and r:
            entries.append(_td_entry("Rushing", team, opp_team, rb_id, r["player_display_name"], t))
        rr = project_count(models["receiving_yards"], rb_id, opp_team, env, with_ladder=True)
        if rr:
            entries.append(_prop_entry("Receiving", "Receiving Yds", team, opp_team, rb_id, rr, ladder=True))
        rec = project_count(models["receptions"], rb_id, opp_team, env)
        if rec:
            entries.append(_prop_entry("Receiving", "Receptions", team, opp_team, rb_id, rec))

    for wrte_pos in ("WR", "TE"):
        for pid in picks.get(wrte_pos, []):
            if is_out(pid):
                continue
            r = project_count(models["receiving_yards"], pid, opp_team, env, with_ladder=True)
            if r:
                entries.append(_prop_entry("Receiving", "Receiving Yds", team, opp_team, pid, r, ladder=True))
            rec = project_count(models["receptions"], pid, opp_team, env)
            if rec:
                entries.append(_prop_entry("Receiving", "Receptions", team, opp_team, pid, rec))
            t = project_td(models["td"], pid, opp_team, env)
            if t and r:
                entries.append(_td_entry("Receiving", team, opp_team, pid, r["player_display_name"], t))

    # Anyone left is playing as far as the report knows, but Questionable
    # and Doubtful still carry real risk -- surfaced, not silently dropped.
    for e in entries:
        st = status_for(e.get("player_id"))
        if st in INJURY_BADGE:
            e["injury_status"] = st

    return entries


def env_fill_values(games_df):
    outdoor = games_df[games_df["roof"].isin(["outdoors", "open"])]
    home_implied = games_df["total_line"] / 2 + games_df["spread_line"] / 2
    away_implied = games_df["total_line"] / 2 - games_df["spread_line"] / 2
    implied_fill = float(pd.concat([home_implied, away_implied]).median())
    return float(outdoor["temp"].median()), float(outdoor["wind"].median()), implied_fill


def build_env(row, temp_fill, wind_fill, own_rest, is_home, implied_fill):
    is_dome = 1.0 if row.roof in ("dome", "closed") else 0.0
    temp = 70.0 if is_dome else (temp_fill if pd.isna(row.temp) else float(row.temp))
    wind = 0.0 if is_dome else (wind_fill if pd.isna(row.wind) else float(row.wind))
    if pd.isna(row.total_line) or pd.isna(row.spread_line):
        implied_team_total = implied_fill
    else:
        implied_team_total = float(row.total_line) / 2 + (float(row.spread_line) / 2 if is_home else -float(row.spread_line) / 2)
    return {"is_dome": is_dome, "temp": temp, "wind": wind, "own_rest": float(own_rest),
            "implied_team_total": implied_team_total}


def snapshot_path(season, week, away, home):
    return RESULTS_DIR / f"nfl_{season}_wk{week:02d}_{away}_{home}.json"


def write_prediction_snapshot(season, week, game):
    """Freezes this game's pregame prediction (win probs + full props array)
    the first time it's generated. Never overwritten on later runs, so it
    stays the model's true pregame call even as later refreshes update
    depth charts/trailing stats -- grade_results.py fills in the actual
    outcome once the game is in the books."""
    path = snapshot_path(season, week, game["awayAbbr"], game["homeAbbr"])
    if path.exists():
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "season": season, "week": week,
        "awayAbbr": game["awayAbbr"], "homeAbbr": game["homeAbbr"],
        "awayName": game["awayName"], "homeName": game["homeName"],
        "gameday": game["gameday"],
        "predicted_at": datetime.datetime.now().isoformat(timespec="seconds"),
        # Which over/under machinery produced model_over_prob in this
        # snapshot. Snapshots are frozen forever, so any future NFL track
        # record or calibration chart has to filter on this rather than
        # pooling probabilities from different models (the same trap MLB's
        # PROP_PROB_MODEL guards against).
        "prop_prob_model": PROP_PROB_MODEL,
        "elo_home_prob": game["elo_home_prob"],
        "market_home_prob": game["market_home_prob"],
        "props_snapshot": game["props"],
        "graded": False,
        "actual": None,
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


def rebuild_results_manifest():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in RESULTS_DIR.glob("nfl_*.json") if p.name != "manifest.json")
    with open(RESULTS_DIR / "manifest.json", "w") as f:
        json.dump({"files": files}, f, indent=2)
    return len(files)


def build_nfl_standings():
    """Real NFL standings by division, regular-season games only. Falls back
    to the most recent season with any completed games when the target
    season hasn't started yet (confirmed real: as of this build, 2026 has
    zero completed games in player_stats/schedules -- the whole NFL tab is a
    genuine future-week projection, not a replay) -- labeled with the real
    season year and whether it's a final record, never silently implied to
    be the live current season when it isn't."""
    games = load_games()  # completed games only
    display_season = int(games["season"].max())
    reg = games[(games["season"] == display_season) & (games["game_type"] == "REG")]

    raw_sched = pl.read_parquet(DATA_DIR / "schedules.parquet").to_pandas()
    total_reg_scheduled = len(raw_sched[(raw_sched["season"] == display_season) & (raw_sched["game_type"] == "REG")])
    is_final = len(reg) >= total_reg_scheduled

    teams = pl.read_parquet(DATA_DIR / "teams.parquet").to_pandas()
    conf_div = teams.drop_duplicates("team_abbr").set_index("team_abbr")[["team_conf", "team_division"]]

    records = {}
    for _, r in reg.iterrows():
        home, away = r["home_team"], r["away_team"]
        records.setdefault(home, {"wins": 0, "losses": 0, "ties": 0})
        records.setdefault(away, {"wins": 0, "losses": 0, "ties": 0})
        if r["home_win"] == 1.0:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif r["home_win"] == 0.0:
            records[away]["wins"] += 1
            records[home]["losses"] += 1
        else:
            records[home]["ties"] += 1
            records[away]["ties"] += 1

    rows = []
    for team, rec in records.items():
        if team not in conf_div.index:
            continue
        w, l, t = rec["wins"], rec["losses"], rec["ties"]
        played = w + l + t
        rows.append({
            "team": team, "division": conf_div.loc[team, "team_division"],
            "wins": w, "losses": l, "ties": t,
            "win_pct": round((w + 0.5 * t) / played, 3) if played else 0.0,
        })

    by_division = {}
    for r in rows:
        by_division.setdefault(r["division"], []).append(r)
    for div_rows in by_division.values():
        div_rows.sort(key=lambda r: -r["win_pct"])
        for i, r in enumerate(div_rows):
            r["rank"] = i + 1

    return {"season": display_season, "is_final": is_final, "standings": by_division}


def build_nfl_stat_leaders(top_n=5):
    """Real season-to-date leaders in the major counting stats, same season
    (and same real-completed-games fallback) as build_nfl_standings, straight
    sums from nflreadpy's own player_stats -- no separate data source."""
    stats = pl.read_parquet(DATA_DIR / "player_stats.parquet").to_pandas()
    stats = stats[stats["season_type"] == "REG"]
    display_season = int(stats["season"].max())
    season_stats = stats[stats["season"] == display_season]

    agg = season_stats.groupby(["player_display_name", "team"], as_index=False)[
        ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]
    ].sum()

    def top_leaders(stat_col, label):
        top = agg.sort_values(stat_col, ascending=False).head(top_n)
        return {"stat": label, "leaders": [
            {"player": r["player_display_name"], "team": r["team"], "value": int(r[stat_col])}
            for _, r in top.iterrows() if r[stat_col] > 0
        ]}

    return {
        "season": display_season,
        "leaders": [
            top_leaders("passing_yards", "Passing Yards"),
            top_leaders("passing_tds", "Passing TDs"),
            top_leaders("rushing_yards", "Rushing Yards"),
            top_leaders("rushing_tds", "Rushing TDs"),
            top_leaders("receiving_yards", "Receiving Yards"),
            top_leaders("receiving_tds", "Receiving TDs"),
        ],
    }


def _nfl_reg_games(max_season=None):
    """One row per real REG game (played and unplayed), every season on
    record (or through max_season if given) -- schedules.parquet already
    carries the full season's real schedule, games not yet played included,
    just never used for anything before now."""
    raw = pl.read_parquet(DATA_DIR / "schedules.parquet")
    reg = raw.filter(pl.col("game_type") == "REG")
    if max_season is not None:
        reg = reg.filter(pl.col("season") <= max_season)
    df = reg.select(["season", "week", "home_team", "away_team", "home_score", "away_score"]).to_pandas()
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df.sort_values(["season", "week"]).reset_index(drop=True)


def build_nfl_title_odds():
    """Division title + playoff berth odds, from Monte Carlo simulating the
    real remaining regular-season schedule from each team's current real Elo
    rating -- pipeline/common/season_sim.py, validated in
    pipeline/nfl/backtest_season_sim.py against 6 real completed NFL seasons
    (beats a naive win-rate-extrapolation baseline in 12/12 backtested
    snapshots; pooled division-title Brier 0.1063 vs. a naive equal-
    probability baseline's 0.1875). Division-title rates are shrunk toward
    the field (shrinkage=0.90, NFL's own backtested value); playoff-berth
    rates ship as the simulator's raw output (not separately calibration-
    checked). Targets the real current season from schedules.parquet
    (whether it's started yet or not) rather than build_nfl_standings'
    most-recent-completed fallback -- unlike standings, there's a genuine
    remaining schedule to simulate even at 0 games played."""
    from pipeline.nfl.elo_model import run_elo as _nfl_run_elo
    from pipeline.common.season_sim import simulate_remaining_wins, division_title_rates, playoff_berth_rates, shrink_toward_field

    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    all_games = _nfl_reg_games()
    target_season = int(all_games["season"].max())
    played = all_games[all_games["home_score"].notna()].reset_index(drop=True)
    remaining = all_games[(all_games["season"] == target_season) & all_games["home_score"].isna()][["home_team", "away_team"]]

    _, ratings = _nfl_run_elo(played, k=elo_params["k"], home_adv=elo_params["home_adv"],
                              scale=elo_params["scale"], rest_adv=elo_params.get("rest_adv", 0.0),
                              season_regression=elo_params["season_regression"], return_ratings=True)

    current_wins = {}
    for _, r in played[played["season"] == target_season].iterrows():
        current_wins.setdefault(r["home_team"], 0)
        current_wins.setdefault(r["away_team"], 0)
        if r["home_win"] == 1.0:
            current_wins[r["home_team"]] += 1
        else:
            current_wins[r["away_team"]] += 1

    teams_df = pl.read_parquet(DATA_DIR / "teams.parquet").to_pandas().drop_duplicates("team_abbr")
    conf_div = teams_df.set_index("team_abbr")[["team_conf", "team_division"]]

    # ratings (run across all real history back to 2019) still carries a
    # stale entry for any relocated/renamed franchise code (e.g. OAK before
    # its LV rename) that never gets touched again after the move -- and
    # teams.parquet keeps historical duplicate rows for the same reason
    # (LA/LAR, LAC/SD, LV/OAK). A target season's own real schedule can only
    # ever reference currently-active codes, so it's the one reliable source
    # for "which 32 teams actually exist right now" to filter both against.
    season_teams = set(all_games[all_games["season"] == target_season]["home_team"]) | \
                   set(all_games[all_games["season"] == target_season]["away_team"])
    ratings = {t: r for t, r in ratings.items() if t in season_teams}

    teams, sim_wins = simulate_remaining_wins(remaining, ratings, elo_params["home_adv"], elo_params["scale"], n_sims=5000)
    base_wins = np.array([current_wins.get(t, 0) for t in teams])
    final_wins = base_wins[None, :] + sim_wins

    team_divisions = {t: conf_div.loc[t, "team_division"] for t in teams if t in conf_div.index}
    div_rates = division_title_rates(teams, final_wins, team_divisions)
    for team_rates in div_rates.values():
        for t in team_rates:
            team_rates[t] = round(shrink_toward_field(team_rates[t], 4, 0.90), 4)

    conferences = {t: conf_div.loc[t, "team_conf"] for t in teams if t in conf_div.index}
    playoff_rates = playoff_berth_rates(teams, final_wins, conferences, berths_per_conference=7,
                                        division_winners_guaranteed=team_divisions, top_n_per_division=1)

    return {"division_title_pct": div_rates, "playoff_pct": playoff_rates}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None, help="Override auto-detected target season")
    args = parser.parse_args()

    if args.season is not None:
        target_season = args.season
        current_week = 1
    else:
        target_season, current_week = detect_target_week()

    names = team_names()
    games_df = load_games()
    team_stats_table = build_team_stats_table()
    season_sched = get_season_schedule(target_season)
    all_weeks = sorted(season_sched["week"].unique().tolist())
    print(f"Season {target_season}: generating weeks {all_weeks[0]}-{all_weeks[-1]}, current={current_week}")

    elo_preds, elo_params = elo_predictions_for_season(games_df, season_sched)
    starters, depth_chart_dt = get_starters(target_season)
    print(f"Depth charts as of {depth_chart_dt}", flush=True)
    temp_fill, wind_fill, implied_fill = env_fill_values(games_df)

    print("Fitting prop models (once, reused across all weeks)...", flush=True)
    # Positions and per-stat distribution/volume config all come from
    # PROP_CONFIG so the generator and the backtest cannot drift apart.
    prop_models = {stat: prepare_count_model(stat, cfg["positions"])
                   for stat, cfg in PROP_CONFIG.items()}
    prop_models["td"] = prepare_td_model()
    print("Prop models ready.", flush=True)

    injury_status = load_injury_status(target_season)

    print("Fetching current DraftKings game lines (one bulk call)...", flush=True)
    odds_map = fetch_current_week_odds_map(names)
    print(f"DK current lines available for {len(odds_map)} games.", flush=True)

    weeks_out = {}
    for week in all_weeks:
        week_rows = season_sched[season_sched["week"] == week].reset_index(drop=True)
        week_elo = [elo_preds[i] for i in season_sched.index[season_sched["week"] == week]]

        games_out = []
        for i, row in enumerate(week_rows.itertuples(index=False)):
            away, home = row.away_team, row.home_team
            already_played = pd.notna(getattr(row, "home_score", None))

            props = []
            if not already_played:
                away_env = build_env(row, temp_fill, wind_fill, row.away_rest, is_home=False, implied_fill=implied_fill)
                home_env = build_env(row, temp_fill, wind_fill, row.home_rest, is_home=True, implied_fill=implied_fill)
                props = (build_props_for_team(away, home, starters, away_env, prop_models,
                                              injuries=injury_status, week=week)
                         + build_props_for_team(home, away, starters, home_env, prop_models,
                                                injuries=injury_status, week=week))

            elo_p = week_elo[i]
            market_home_prob = round(float(row.market_home_prob), 4) if pd.notna(row.market_home_prob) else None
            elo_home_prob = round(float(elo_p), 4) if elo_p is not None else None

            # "Good value" (pregame odds only, per spec): does the model's win
            # probability for a side beat that side's own pregame fair %?
            good_value_home = good_value_away = None
            if market_home_prob is not None and elo_home_prob is not None:
                good_value_home = elo_home_prob > market_home_prob
                good_value_away = (1 - elo_home_prob) > (1 - market_home_prob)

            games_out.append({
                "awayAbbr": away, "homeAbbr": home,
                "awayName": names.get(away, away), "homeName": names.get(home, home),
                "gameday": row.gameday, "weekday": row.weekday, "gametime": row.gametime,
                "primetime": primetime_label(row.weekday, row.gametime),
                "stadium": row.stadium if pd.notna(row.stadium) else None,
                "location": row.location if pd.notna(row.location) else None,
                "spread_line": row.spread_line if pd.notna(row.spread_line) else None,
                "total_line": row.total_line if pd.notna(row.total_line) else None,
                "mlAway": int(row.away_moneyline) if pd.notna(row.away_moneyline) else None,
                "mlHome": int(row.home_moneyline) if pd.notna(row.home_moneyline) else None,
                "market_home_prob": market_home_prob,
                "good_value_home": good_value_home,
                "good_value_away": good_value_away,
                "elo_home_prob": elo_home_prob,
                "roof": row.roof if pd.notna(row.roof) else None,
                "away_rest": int(row.away_rest) if pd.notna(row.away_rest) else None,
                "home_rest": int(row.home_rest) if pd.notna(row.home_rest) else None,
                "awayTeamStats": current_team_stats(team_stats_table, away),
                "homeTeamStats": current_team_stats(team_stats_table, home),
                "already_played": bool(already_played),
                "props": props,
            })

        attach_current_lines(games_out, names, odds_map)
        if week == current_week:
            try:
                attach_td_odds(games_out, names, odds_map)
                n_with_td = sum(1 for g in games_out for p in g["props"] if p["market"] == "Anytime TD" and "dk_odds" in p)
                print(f"  week {week}: attached real DK TD odds to {n_with_td} player props", flush=True)
            except Exception as e:
                print(f"  week {week}: TD odds attach failed, continuing without them: {e}", flush=True)

        # Snapshot only the current week: freezing every future week's props
        # this far ahead would lock in a July depth chart for a December
        # game, which will be badly stale by the time it's actually played.
        # Each week gets snapshotted exactly once, right before it happens,
        # as current_week advances.
        if week == current_week:
            for g in games_out:
                if not g["already_played"]:
                    write_prediction_snapshot(target_season, week, g)

        weeks_out[str(week)] = {"games": games_out}
        print(f"  week {week}: {len(games_out)} games, "
              f"{sum(1 for g in games_out if g['market_home_prob'] is not None)} with market odds", flush=True)

    n_snapshots = rebuild_results_manifest()
    print(f"Results manifest: {n_snapshots} prediction snapshots on disk.", flush=True)

    print("Building NFL standings + stat leaders...")
    nfl_title_odds = build_nfl_title_odds()
    record_title_odds("nfl", nfl_title_odds, season=target_season)
    payload = {
        "season": target_season, "current_week": current_week,
        "elo_params": elo_params,
        "depth_chart_as_of": str(depth_chart_dt),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "weeks": weeks_out,
        "season_info": {"standings": build_nfl_standings(), "stat_leaders": build_nfl_stat_leaders(),
                        "title_odds": nfl_title_odds},
    }
    out_path = DATA_DIR / "dashboard_current_week.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

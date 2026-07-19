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
from pipeline.nfl.props.prop_data import build_prop_table
from pipeline.nfl.props.current_state import player_current_trailing, defense_current_trailing
from pipeline.nfl.props.prop_models import FEATURES, yardage_over_prob
from pipeline.nfl.props.nfl_td_odds import attach_td_odds
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from scipy.stats import norm

DATA_DIR = ROOT / "data" / "nfl"


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


def yardage_ladder(pred_mean, resid_std, own_avg):
    """Lines in steps of 10 around the player's current trailing average,
    e.g. 190/200/210/220/230 for a receiver averaging ~207 yds/game."""
    base = max(round(own_avg / 10) * 10, 10)
    ladder = []
    for off in YARDAGE_LADDER_OFFSETS:
        line = base + off
        if line <= 0:
            continue
        ladder.append({"line": float(line), "over_prob": round(float(yardage_over_prob(pred_mean, resid_std, line)), 3)})
    return ladder


def prepare_count_model(stat_col, positions):
    """Fit a RidgeCV model ONCE on full history for any continuous/count stat
    (yards, completions, attempts, carries, receptions...). Doesn't depend on
    the target week, so every game/week across the whole season reuses it."""
    hist = build_prop_table(stat_col, positions)
    model = RidgeCV(alphas=np.logspace(-1, 3, 25))
    model.fit(hist[FEATURES].values, hist["actual"].values)
    resid_std = max(float(np.std(hist["actual"].values - model.predict(hist[FEATURES].values))), 1e-6)
    return {
        "model": model, "resid_std": resid_std,
        "own": player_current_trailing(stat_col, positions),
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
    feat_row = [[own_avg, opp_avg, env["is_dome"], env["temp"], env["wind"], env["own_rest"]]]
    pred_mean = float(prep["model"].predict(feat_row)[0])
    line = round(own_avg * 2) / 2
    over_prob = float(yardage_over_prob(pred_mean, prep["resid_std"], line))
    out = {
        "line": line, "projected": round(pred_mean, 1), "model_over_prob": round(over_prob, 3),
        "player_display_name": own.loc[player_id, "player_display_name"],
        "games_played": int(own.loc[player_id, "games_played"]),
    }
    if with_ladder:
        out["ladder"] = yardage_ladder(pred_mean, prep["resid_std"], own_avg)
    return out


def prepare_td_model():
    hist = build_prop_table("anytime_td", ["RB", "WR", "TE"])
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(hist[FEATURES].values, hist["actual"].values)
    return {
        "model": model,
        "own": player_current_trailing("anytime_td", ["RB", "WR", "TE"]),
        "defense": defense_current_trailing("anytime_td", ["RB", "WR", "TE"]),
    }


def project_td(prep, player_id, opp_team, env):
    own, defense = prep["own"], prep["defense"]
    if player_id not in own.index or opp_team not in defense.index:
        return None

    own_avg = float(own.loc[player_id, "current_avg"])
    opp_avg = float(defense.loc[opp_team])
    if pd.isna(own_avg) or pd.isna(opp_avg):
        return None
    feat_row = [[own_avg, opp_avg, env["is_dome"], env["temp"], env["wind"], env["own_rest"]]]
    prob = float(prep["model"].predict_proba(feat_row)[:, 1][0])
    return {"model_prob": round(prob, 3), "games_played": int(own.loc[player_id, "games_played"])}


def _prop_entry(section, market, team, r, ladder=False):
    e = {"section": section, "player": r["player_display_name"], "team": team, "market": market,
         "line": r["line"], "projected": r["projected"], "model_over_prob": r["model_over_prob"]}
    if ladder and r.get("ladder"):
        e["ladder"] = r["ladder"]
    return e


def _td_entry(section, team, player_name, t):
    return {"section": section, "player": player_name, "team": team, "market": "Anytime TD", "model_prob": t["model_prob"]}


def build_props_for_team(team, opp_team, starters, env, models):
    entries = []
    picks = starters.get(team, {})

    for qb_id in picks.get("QB", []):
        r = project_count(models["passing_yards"], qb_id, opp_team, env, with_ladder=True)
        if r:
            entries.append(_prop_entry("Passing", "Passing Yds", team, r, ladder=True))
        rt = project_count(models["passing_tds"], qb_id, opp_team, env)
        if rt:
            entries.append(_prop_entry("Passing", "Passing TDs", team, rt))
        rc = project_count(models["completions"], qb_id, opp_team, env)
        if rc:
            entries.append(_prop_entry("Passing", "Completions", team, rc))
        ra = project_count(models["attempts"], qb_id, opp_team, env)
        if ra:
            entries.append(_prop_entry("Passing", "Pass Attempts", team, ra))

    for rb_id in picks.get("RB", []):
        r = project_count(models["rushing_yards"], rb_id, opp_team, env, with_ladder=True)
        if r:
            entries.append(_prop_entry("Rushing", "Rushing Yds", team, r, ladder=True))
        rc = project_count(models["carries"], rb_id, opp_team, env)
        if rc:
            entries.append(_prop_entry("Rushing", "Carries", team, rc))
        t = project_td(models["td"], rb_id, opp_team, env)
        if t and r:
            entries.append(_td_entry("Rushing", team, r["player_display_name"], t))
        rr = project_count(models["receiving_yards"], rb_id, opp_team, env, with_ladder=True)
        if rr:
            entries.append(_prop_entry("Receiving", "Receiving Yds", team, rr, ladder=True))
        rec = project_count(models["receptions"], rb_id, opp_team, env)
        if rec:
            entries.append(_prop_entry("Receiving", "Receptions", team, rec))

    for wrte_pos in ("WR", "TE"):
        for pid in picks.get(wrte_pos, []):
            r = project_count(models["receiving_yards"], pid, opp_team, env, with_ladder=True)
            if r:
                entries.append(_prop_entry("Receiving", "Receiving Yds", team, r, ladder=True))
            rec = project_count(models["receptions"], pid, opp_team, env)
            if rec:
                entries.append(_prop_entry("Receiving", "Receptions", team, rec))
            t = project_td(models["td"], pid, opp_team, env)
            if t and r:
                entries.append(_td_entry("Receiving", team, r["player_display_name"], t))

    return entries


def env_fill_values(games_df):
    outdoor = games_df[games_df["roof"].isin(["outdoors", "open"])]
    return float(outdoor["temp"].median()), float(outdoor["wind"].median())


def build_env(row, temp_fill, wind_fill, own_rest):
    is_dome = 1.0 if row.roof in ("dome", "closed") else 0.0
    temp = 70.0 if is_dome else (temp_fill if pd.isna(row.temp) else float(row.temp))
    wind = 0.0 if is_dome else (wind_fill if pd.isna(row.wind) else float(row.wind))
    return {"is_dome": is_dome, "temp": temp, "wind": wind, "own_rest": float(own_rest)}


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
    season_sched = get_season_schedule(target_season)
    all_weeks = sorted(season_sched["week"].unique().tolist())
    print(f"Season {target_season}: generating weeks {all_weeks[0]}-{all_weeks[-1]}, current={current_week}")

    elo_preds, elo_params = elo_predictions_for_season(games_df, season_sched)
    starters, depth_chart_dt = get_starters(target_season)
    print(f"Depth charts as of {depth_chart_dt}", flush=True)
    temp_fill, wind_fill = env_fill_values(games_df)

    print("Fitting prop models (once, reused across all weeks)...", flush=True)
    prop_models = {
        "passing_yards": prepare_count_model("passing_yards", ["QB"]),
        "passing_tds": prepare_count_model("passing_tds", ["QB"]),
        "completions": prepare_count_model("completions", ["QB"]),
        "attempts": prepare_count_model("attempts", ["QB"]),
        "rushing_yards": prepare_count_model("rushing_yards", ["RB"]),
        "carries": prepare_count_model("carries", ["RB"]),
        "receiving_yards": prepare_count_model("receiving_yards", ["RB", "WR", "TE"]),
        "receptions": prepare_count_model("receptions", ["RB", "WR", "TE"]),
        "td": prepare_td_model(),
    }
    print("Prop models ready.", flush=True)

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
                away_env = build_env(row, temp_fill, wind_fill, row.away_rest)
                home_env = build_env(row, temp_fill, wind_fill, row.home_rest)
                props = (build_props_for_team(away, home, starters, away_env, prop_models)
                         + build_props_for_team(home, away, starters, home_env, prop_models))

            elo_p = week_elo[i]
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
                "market_home_prob": round(float(row.market_home_prob), 4) if pd.notna(row.market_home_prob) else None,
                "elo_home_prob": round(float(elo_p), 4) if elo_p is not None else None,
                "roof": row.roof if pd.notna(row.roof) else None,
                "away_rest": int(row.away_rest) if pd.notna(row.away_rest) else None,
                "home_rest": int(row.home_rest) if pd.notna(row.home_rest) else None,
                "props": props,
            })
        if week == current_week:
            try:
                attach_td_odds(games_out, names)
                n_with_td = sum(1 for g in games_out for p in g["props"] if p["market"] == "Anytime TD" and "dk_odds" in p)
                print(f"  week {week}: attached real DK TD odds to {n_with_td} player props", flush=True)
            except Exception as e:
                print(f"  week {week}: TD odds attach failed, continuing without them: {e}", flush=True)

        weeks_out[str(week)] = {"games": games_out}
        print(f"  week {week}: {len(games_out)} games, "
              f"{sum(1 for g in games_out if g['market_home_prob'] is not None)} with market odds", flush=True)

    payload = {
        "season": target_season, "current_week": current_week,
        "elo_params": elo_params,
        "depth_chart_as_of": str(depth_chart_dt),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "weeks": weeks_out,
    }
    out_path = DATA_DIR / "dashboard_current_week.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

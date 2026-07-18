"""Generate REAL projections for an upcoming, not-yet-played NFL week:
- Elo win probability, ratings carried forward from all completed games
  through the fitted hyperparameters (no re-fitting on the future week).
- Real posted opening market lines (moneyline/spread/total), already present
  in nflverse schedules for the upcoming season.
- Player props for each team's current depth-chart starters (QB/RB1/WR1),
  using their actual current trailing rate vs. the opponent defense's actual
  current trailing allowed-rate, fit on the FULL historical dataset (the
  target week is genuinely in the future, so there's no leakage to guard
  against the way there is in a backtest).
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


def get_target_week_schedule(target_season, target_week):
    import nflreadpy as nfl
    sched = nfl.load_schedules(seasons=[target_season]).to_pandas()
    wk = sched[sched["week"] == target_week].copy()
    wk["market_home_prob_raw_away"] = moneyline_to_prob(wk["away_moneyline"].values)
    wk["market_home_prob_raw_home"] = moneyline_to_prob(wk["home_moneyline"].values)
    overround = wk["market_home_prob_raw_away"] + wk["market_home_prob_raw_home"]
    wk["market_home_prob"] = wk["market_home_prob_raw_home"] / overround
    return wk


def elo_predictions_for_target(games_df, target_wk):
    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    future_rows = pd.DataFrame({
        "season": target_wk["season"].values,
        "week": target_wk["week"].values,
        "home_team": target_wk["home_team"].values,
        "away_team": target_wk["away_team"].values,
        "margin": np.nan,
        "home_win": np.nan,
        "location": target_wk["location"].values,
        "home_rest": target_wk["home_rest"].values,
        "away_rest": target_wk["away_rest"].values,
    })
    cols = ["season", "week", "home_team", "away_team", "margin", "home_win", "location", "home_rest", "away_rest"]
    combined = pd.concat([games_df[cols], future_rows], ignore_index=True)
    preds = run_elo(combined, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"],
                    rest_adv=elo_params.get("rest_adv", 0.0), season_regression=elo_params.get("season_regression", 0.75))
    n_future = len(future_rows)
    return preds[-n_future:], elo_params


def get_starters(target_season):
    import nflreadpy as nfl
    dc = nfl.load_depth_charts(seasons=[target_season]).to_pandas()
    latest_dt = dc["dt"].max()
    dc = dc[dc["dt"] == latest_dt]

    starters = {}  # team -> {"QB": gsis_id, "RB": gsis_id, "WR": gsis_id}
    for team, grp in dc.groupby("team"):
        picks = {}
        for pos in ("QB", "RB", "WR"):
            row = grp[(grp["pos_abb"] == pos) & (grp["pos_rank"] == 1)]
            if len(row) and pd.notna(row.iloc[0]["gsis_id"]):
                picks[pos] = row.iloc[0]["gsis_id"]
        starters[team] = picks
    return starters, latest_dt


def fit_and_project_yardage(stat_col, positions, player_id, opp_team, env):
    hist = build_prop_table(stat_col, positions)
    model = RidgeCV(alphas=np.logspace(-1, 3, 25))
    model.fit(hist[FEATURES].values, hist["actual"].values)
    resid_std = float(np.std(hist["actual"].values - model.predict(hist[FEATURES].values)))

    own = player_current_trailing(stat_col, positions)
    defense = defense_current_trailing(stat_col, positions)
    if player_id not in own.index or opp_team not in defense.index:
        return None

    own_avg = float(own.loc[player_id, "current_avg"])
    opp_avg = float(defense.loc[opp_team])
    feat_row = [[own_avg, opp_avg, env["is_dome"], env["temp"], env["wind"], env["own_rest"]]]
    pred_mean = float(model.predict(feat_row)[0])
    line = round(own_avg * 2) / 2
    over_prob = float(yardage_over_prob(pred_mean, max(resid_std, 1e-6), line))
    return {
        "line": line, "projected": round(pred_mean, 1), "model_over_prob": round(over_prob, 3),
        "player_display_name": own.loc[player_id, "player_display_name"],
        "games_played": int(own.loc[player_id, "games_played"]),
    }


def fit_and_project_td(player_id, opp_team, env):
    hist = build_prop_table("anytime_td", ["RB", "WR", "TE"])
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(hist[FEATURES].values, hist["actual"].values)

    own = player_current_trailing("anytime_td", ["RB", "WR", "TE"])
    defense = defense_current_trailing("anytime_td", ["RB", "WR", "TE"])
    if player_id not in own.index or opp_team not in defense.index:
        return None

    own_avg = float(own.loc[player_id, "current_avg"])
    opp_avg = float(defense.loc[opp_team])
    feat_row = [[own_avg, opp_avg, env["is_dome"], env["temp"], env["wind"], env["own_rest"]]]
    prob = float(model.predict_proba(feat_row)[:, 1][0])
    return {"model_prob": round(prob, 3), "games_played": int(own.loc[player_id, "games_played"])}


def build_props_for_team(team, opp_team, starters, env):
    entries = []
    picks = starters.get(team, {})

    if "QB" in picks:
        r = fit_and_project_yardage("passing_yards", ["QB"], picks["QB"], opp_team, env)
        if r:
            entries.append({"player": r["player_display_name"], "team": team, "market": "Passing Yds",
                             "line": r["line"], "projected": r["projected"], "model_over_prob": r["model_over_prob"]})

    if "RB" in picks:
        r = fit_and_project_yardage("rushing_yards", ["RB"], picks["RB"], opp_team, env)
        if r:
            entries.append({"player": r["player_display_name"], "team": team, "market": "Rushing Yds",
                             "line": r["line"], "projected": r["projected"], "model_over_prob": r["model_over_prob"]})
        t = fit_and_project_td(picks["RB"], opp_team, env)
        if t:
            entries.append({"player": r["player_display_name"] if r else None, "team": team, "market": "Anytime TD",
                             "model_prob": t["model_prob"]})

    if "WR" in picks:
        r = fit_and_project_yardage("receiving_yards", ["WR", "TE"], picks["WR"], opp_team, env)
        if r:
            entries.append({"player": r["player_display_name"], "team": team, "market": "Receiving Yds",
                             "line": r["line"], "projected": r["projected"], "model_over_prob": r["model_over_prob"]})
        t = fit_and_project_td(picks["WR"], opp_team, env)
        if t:
            entries.append({"player": r["player_display_name"] if r else None, "team": team, "market": "Anytime TD",
                             "model_prob": t["model_prob"]})

    return [e for e in entries if e.get("player")]


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
    parser.add_argument("--week", type=int, default=None, help="Override auto-detected target week")
    args = parser.parse_args()

    if args.season is not None and args.week is not None:
        target_season, target_week = args.season, args.week
    else:
        target_season, target_week = detect_target_week()

    names = team_names()
    games_df = load_games()
    target_wk = get_target_week_schedule(target_season, target_week)
    print(f"Target: {target_season} week {target_week} -- {len(target_wk)} games")

    elo_preds, elo_params = elo_predictions_for_target(games_df, target_wk)
    starters, depth_chart_dt = get_starters(target_season)
    print(f"Depth charts as of {depth_chart_dt}")
    temp_fill, wind_fill = env_fill_values(games_df)

    games_out = []
    for i, row in enumerate(target_wk.itertuples(index=False)):
        away, home = row.away_team, row.home_team
        away_env = build_env(row, temp_fill, wind_fill, row.away_rest)
        home_env = build_env(row, temp_fill, wind_fill, row.home_rest)
        props = build_props_for_team(away, home, starters, away_env) + build_props_for_team(home, away, starters, home_env)
        games_out.append({
            "awayAbbr": away, "homeAbbr": home,
            "awayName": names.get(away, away), "homeName": names.get(home, home),
            "gameday": row.gameday,
            "spread_line": row.spread_line, "total_line": row.total_line,
            "mlAway": int(row.away_moneyline), "mlHome": int(row.home_moneyline),
            "market_home_prob": round(float(row.market_home_prob), 4),
            "elo_home_prob": round(float(elo_preds[i]), 4),
            "roof": row.roof if pd.notna(row.roof) else None, "away_rest": int(row.away_rest), "home_rest": int(row.home_rest),
            "props": props,
        })
        print(f"  {away} @ {home}: market_home={row.market_home_prob:.3f} elo_home={elo_preds[i]:.3f} props={len(props)}")

    payload = {
        "season": target_season, "week": target_week,
        "elo_params": elo_params,
        "depth_chart_as_of": str(depth_chart_dt),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "games": games_out,
    }
    out_path = DATA_DIR / "dashboard_current_week.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    archive_path = DATA_DIR / f"dashboard_{target_season}_week{target_week}.json"
    with open(archive_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path} (and archived to {archive_path})")


if __name__ == "__main__":
    main()

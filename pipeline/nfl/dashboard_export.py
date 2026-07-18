"""Generate a JSON payload of REAL, fitted, backtested model output for one
historical NFL week, in a shape the existing 'Phins Up' dashboard UI can
render directly -- replacing the hand-typed TEAM_RATING/PLAYER_BASE objects
and guessed formulas in the original prototype.

Because the week chosen already happened, actual results are included too,
so the UI can show the model's calibration directly (predicted vs. actual).
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.games import load_games
from pipeline.nfl.elo_model import run_elo
from pipeline.nfl.props.prop_data import build_prop_table
from pipeline.nfl.props.prop_models import walk_forward_yardage, walk_forward_anytime_td, yardage_over_prob

SEASON = 2024
WEEK = 1
DATA_DIR = ROOT / "data" / "nfl"


def team_names():
    import nflreadpy as nfl
    t = nfl.load_teams().to_pandas()
    return dict(zip(t["team_abbr"], t["team_name"]))


def top_players(season, week):
    ps = pl.read_parquet(DATA_DIR / "player_stats.parquet").to_pandas()
    wk = ps[(ps["season"] == season) & (ps["week"] == week)]

    picks = {}  # (team) -> {"QB": row, "RB": row, "REC": row}
    for team, grp in wk.groupby("team"):
        team_picks = {}
        qbs = grp[grp["position"] == "QB"].sort_values("attempts", ascending=False)
        if len(qbs):
            team_picks["QB"] = qbs.iloc[0]
        rbs = grp[grp["position"] == "RB"].sort_values("carries", ascending=False)
        if len(rbs):
            team_picks["RB"] = rbs.iloc[0]
        recs = grp[grp["position"].isin(["WR", "TE"])].sort_values("targets", ascending=False)
        if len(recs):
            team_picks["REC"] = recs.iloc[0]
        picks[team] = team_picks
    return picks


def yardage_predictions(stat_col, positions):
    df = build_prop_table(stat_col, positions)
    model_pred, resid_std, naive_pred = walk_forward_yardage(df, [SEASON])
    df = df.assign(model_pred=model_pred, resid_std=resid_std, naive_pred=naive_pred)
    return df[(df["season"] == SEASON) & (df["week"] == WEEK)].set_index("player_id")


def td_predictions():
    df = build_prop_table("anytime_td", ["RB", "WR", "TE"])
    model_pred = walk_forward_anytime_td(df, [SEASON])
    df = df.assign(model_pred=model_pred)
    return df[(df["season"] == SEASON) & (df["week"] == WEEK)].set_index("player_id")


def main():
    names = team_names()
    games_df = load_games()

    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]
    elo_preds_full = run_elo(games_df, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"], rest_adv=elo_params.get("rest_adv", 0.0))
    games_df = games_df.assign(elo_home_prob=elo_preds_full)

    week_games = games_df[(games_df["season"] == SEASON) & (games_df["week"] == WEEK)].reset_index(drop=True)

    picks = top_players(SEASON, WEEK)
    pass_pred = yardage_predictions("passing_yards", ["QB"])
    rush_pred = yardage_predictions("rushing_yards", ["RB"])
    rec_pred = yardage_predictions("receiving_yards", ["WR", "TE"])
    td_pred = td_predictions()

    def prop_entries(team):
        entries = []
        tp = picks.get(team, {})

        if "QB" in tp and tp["QB"]["player_id"] in pass_pred.index:
            r = pass_pred.loc[tp["QB"]["player_id"]]
            line = round(r["naive_pred"] * 2) / 2
            entries.append({
                "player": tp["QB"]["player_display_name"], "team": team, "market": "Passing Yds",
                "line": line, "projected": round(float(r["model_pred"]), 1),
                "model_over_prob": round(float(yardage_over_prob(r["model_pred"], r["resid_std"], line)), 3),
                "actual": float(r["actual"]),
            })

        if "RB" in tp and tp["RB"]["player_id"] in rush_pred.index:
            r = rush_pred.loc[tp["RB"]["player_id"]]
            line = round(r["naive_pred"] * 2) / 2
            entries.append({
                "player": tp["RB"]["player_display_name"], "team": team, "market": "Rushing Yds",
                "line": line, "projected": round(float(r["model_pred"]), 1),
                "model_over_prob": round(float(yardage_over_prob(r["model_pred"], r["resid_std"], line)), 3),
                "actual": float(r["actual"]),
            })
            if tp["RB"]["player_id"] in td_pred.index:
                t = td_pred.loc[tp["RB"]["player_id"]]
                entries.append({
                    "player": tp["RB"]["player_display_name"], "team": team, "market": "Anytime TD",
                    "model_prob": round(float(t["model_pred"]), 3), "actual": bool(t["actual"]),
                })

        if "REC" in tp and tp["REC"]["player_id"] in rec_pred.index:
            r = rec_pred.loc[tp["REC"]["player_id"]]
            line = round(r["naive_pred"] * 2) / 2
            entries.append({
                "player": tp["REC"]["player_display_name"], "team": team, "market": "Receiving Yds",
                "line": line, "projected": round(float(r["model_pred"]), 1),
                "model_over_prob": round(float(yardage_over_prob(r["model_pred"], r["resid_std"], line)), 3),
                "actual": float(r["actual"]),
            })
            if tp["REC"]["player_id"] in td_pred.index:
                t = td_pred.loc[tp["REC"]["player_id"]]
                entries.append({
                    "player": tp["REC"]["player_display_name"], "team": team, "market": "Anytime TD",
                    "model_prob": round(float(t["model_pred"]), 3), "actual": bool(t["actual"]),
                })

        return entries

    games_out = []
    for row in week_games.itertuples(index=False):
        games_out.append({
            "awayAbbr": row.away_team, "homeAbbr": row.home_team,
            "awayName": names.get(row.away_team, row.away_team),
            "homeName": names.get(row.home_team, row.home_team),
            "gameday": row.gameday,
            "spread_line": row.spread_line, "total_line": row.total_line,
            "mlAway": int(row.away_moneyline), "mlHome": int(row.home_moneyline),
            "market_home_prob": round(float(row.market_home_prob), 4),
            "elo_home_prob": round(float(row.elo_home_prob), 4),
            "actual_away_score": int(row.away_score), "actual_home_score": int(row.home_score),
            "props": prop_entries(row.away_team) + prop_entries(row.home_team),
        })

    payload = {
        "season": SEASON, "week": WEEK,
        "elo_params": elo_params,
        "games": games_out,
    }

    out_path = DATA_DIR / "dashboard_week_sample.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(games_out)} games to {out_path}")
    for g in games_out:
        print(f"  {g['awayAbbr']} @ {g['homeAbbr']}: market_home={g['market_home_prob']:.3f} elo_home={g['elo_home_prob']:.3f} "
              f"actual={g['actual_away_score']}-{g['actual_home_score']}  props={len(g['props'])}")


if __name__ == "__main__":
    main()

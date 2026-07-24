"""Generate real Elo win-probability predictions for the current NHL week,
using ratings carried forward through every real completed game (2018-19
through 2025-26). Phase 1 dashboard output -- model win probability only,
no player props or real market odds yet (those need their own data pipeline,
same as MLB's later phases).

"Current week" is the Monday-Sunday week containing the earliest date (from
today onward) with a real scheduled regular-season game, not the literal
wall-clock week -- during the off-season (like now: next season starts
2026-09-29) that's the season's actual opening week; once the season is
underway this naturally becomes today's real week, exactly the same
"find the next unplayed game" approach NFL's detect_target_week uses rather
than assuming the literal current calendar date always has games."""
import sys
import pathlib
import json
import datetime
import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nhl.games import load_games
from pipeline.nhl.elo_model import run_elo
from pipeline.nhl.team_map import normalize_team

DATA_DIR = ROOT / "data" / "nhl"
SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule"


def _fetch_week(date_str):
    resp = requests.get(f"{SCHEDULE_URL}/{date_str}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def detect_target_date(today):
    """Earliest date (today onward) with a real scheduled regular-season
    game -- steps forward a week at a time (the schedule endpoint's own
    window size), up to a year out."""
    d = pd.Timestamp(today)
    for _ in range(53):
        data = _fetch_week(d.strftime("%Y-%m-%d"))
        for week in data.get("gameWeek", []):
            if any(g.get("gameType") == 2 for g in week.get("games", [])):
                return datetime.date.fromisoformat(week["date"])
        d += pd.Timedelta(days=7)
    raise RuntimeError("No upcoming NHL regular-season games found within a year.")


def week_dates(anchor_date):
    monday = anchor_date - datetime.timedelta(days=anchor_date.weekday())
    return [(monday + datetime.timedelta(days=i)).isoformat() for i in range(7)]


def get_slate_for_date(target_date):
    data = _fetch_week(target_date)
    for week in data.get("gameWeek", []):
        if week["date"] != target_date:
            continue
        out = []
        for g in week.get("games", []):
            if g.get("gameType") != 2:
                continue
            away, home = g["awayTeam"], g["homeTeam"]
            out.append({
                "target_date": target_date,
                "away_team": normalize_team(away["abbrev"]), "home_team": normalize_team(home["abbrev"]),
                "away_name": f"{away['placeName']['default']} {away['commonName']['default']}",
                "home_name": f"{home['placeName']['default']} {home['commonName']['default']}",
                "game_datetime": g.get("startTimeUTC"),
                "already_played": g.get("gameState") in ("OFF", "FINAL"),
                "away_score": away.get("score"), "home_score": home.get("score"),
            })
        return out
    return []


def current_team_scoring_rates(games_df):
    """Each team's latest known trailing goals-scored/goals-allowed (the
    same columns games.py already computes, home_trailing_goals_scored
    etc.) -- a plain trailing-average-based projected total, same honesty
    framing as MLB's version of this: not a competing win-probability
    model, not separately backtested."""
    home = games_df[["game_date", "home_team", "home_trailing_goals_scored", "home_trailing_goals_allowed"]].rename(
        columns={"home_team": "team", "home_trailing_goals_scored": "scored", "home_trailing_goals_allowed": "allowed"})
    away = games_df[["game_date", "away_team", "away_trailing_goals_scored", "away_trailing_goals_allowed"]].rename(
        columns={"away_team": "team", "away_trailing_goals_scored": "scored", "away_trailing_goals_allowed": "allowed"})
    long = pd.concat([home, away], ignore_index=True).sort_values("game_date")
    return long.groupby("team").tail(1).set_index("team")[["scored", "allowed"]]


def projected_total(rates, home_team, away_team):
    if home_team not in rates.index or away_team not in rates.index:
        return None
    h, a = rates.loc[home_team], rates.loc[away_team]
    if pd.isna(h["scored"]) or pd.isna(h["allowed"]) or pd.isna(a["scored"]) or pd.isna(a["allowed"]):
        return None
    home_exp = (h["scored"] + a["allowed"]) / 2
    away_exp = (a["scored"] + h["allowed"]) / 2
    return round(float(home_exp + away_exp), 1)


def elo_predictions(games_df, slate):
    with open(ROOT / "notebooks_out" / "nhl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    # season = current season's start year (not used for MOV/outcome, only
    # for the between-season regression check in run_elo) -- games this
    # week belong to the season that most recently started.
    current_season = int(games_df["season"].max())
    if pd.Timestamp(slate[0]["target_date"]) >= pd.Timestamp(f"{current_season+1}-09-01"):
        current_season += 1

    future_rows = pd.DataFrame({
        "season": [current_season] * len(slate),
        "game_date": [pd.Timestamp(g["target_date"]) for g in slate],
        "home_team": [g["home_team"] for g in slate],
        "away_team": [g["away_team"] for g in slate],
        "margin": np.nan, "home_win": np.nan,
    })
    cols = ["season", "game_date", "home_team", "away_team", "margin", "home_win"]
    combined = pd.concat([games_df[cols], future_rows], ignore_index=True)
    preds = run_elo(combined, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"],
                     season_regression=elo_params["season_regression"])
    return preds[-len(slate):], elo_params


def build_day_payload(date, games):
    return {
        "date": date, "weekday": datetime.date.fromisoformat(date).strftime("%A"),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "games": games,
    }


def main(today=None):
    today = today or datetime.date.today()

    target_date = detect_target_date(today)
    # The payload's "today" is this anchor date, not the literal wall-clock
    # date -- during the off-season those differ (today has no games; the
    # anchor is the season's actual opening day), and the frontend defaults
    # its day-picker to whatever "today" says, so it needs to be a date that
    # actually exists in `days`. Once the season is underway these two
    # naturally converge, since detect_target_date finds literal today
    # itself whenever today has a real game.
    anchor_iso = target_date.isoformat()
    dates = week_dates(target_date)
    print(f"Anchor date (earliest upcoming game from {today.isoformat()}): {target_date}. Week: {dates[0]} to {dates[-1]}")

    combined_slate = []
    for d in dates:
        raw = get_slate_for_date(d)
        combined_slate.extend(raw)
        print(f"  {d}: {len(raw)} games scheduled")

    days_out = {d: build_day_payload(d, []) for d in dates}
    if not combined_slate:
        _write_payload(dates, anchor_iso, days_out)
        return

    games_df = load_games()
    elo_preds, elo_params = elo_predictions(games_df, combined_slate)
    scoring_rates = current_team_scoring_rates(games_df)

    by_date = {}
    for i, g in enumerate(combined_slate):
        print(f"  {g['target_date']} {g['away_team']} @ {g['home_team']}: model_home={elo_preds[i]:.3f}")
        # Renamed to match the MLB/NFL template's existing field naming
        # convention (awayAbbr/homeAbbr/awayName/homeName/gameDatetime) --
        # the frontend is shared across all three sports' tabs.
        out_game = {
            "awayAbbr": g["away_team"], "homeAbbr": g["home_team"],
            "awayName": g["away_name"], "homeName": g["home_name"],
            "gameDatetime": g["game_datetime"],
            "already_played": g["already_played"],
            "away_score": g["away_score"], "home_score": g["home_score"],
            "elo_home_prob": round(float(elo_preds[i]), 4),
            "model_total_goals": projected_total(scoring_rates, g["home_team"], g["away_team"]),
        }
        by_date.setdefault(g["target_date"], []).append(out_game)

    for d, day_games in by_date.items():
        days_out[d] = build_day_payload(d, day_games)

    _write_payload(dates, anchor_iso, days_out, elo_params)


def _write_payload(dates, today_iso, days_out, elo_params=None):
    payload = {
        "week_start": dates[0], "week_end": dates[-1], "today": today_iso,
        "elo_params": elo_params,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "days": days_out,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "dashboard_current_slate.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    total_games = sum(len(day["games"]) for day in days_out.values())
    print(f"\nWrote {out_path} -- {len(days_out)} days, {total_games} total games")


if __name__ == "__main__":
    main()

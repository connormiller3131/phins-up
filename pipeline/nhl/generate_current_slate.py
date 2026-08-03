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
from pipeline.nhl.goalie_ratings import team_recent_save_pct
from pipeline.common.odds_history import record_title_odds

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


def team_stats_for_dropdown(rates, team):
    """Team Stats dropdown data for NHL. goals for/against comes from the
    schedule endpoint's trailing rates; goalie_save_pct comes from the
    separate real boxscore pull (pull_goalie_games.py) -- shown as plain
    real recent form, not a model input (backtest_goals_model.py found no
    predictive signal there, so projected_score doesn't use it).
    Returns None if the team has no trailing scoring history yet."""
    if team not in rates.index:
        return None
    r = rates.loc[team]
    if pd.isna(r["scored"]) or pd.isna(r["allowed"]):
        return None
    defense = {"goals_allowed_per_game": round(float(r["allowed"]), 2)}
    save_pct = team_recent_save_pct(team)
    if save_pct is not None:
        defense["goalie_save_pct"] = round(save_pct, 3)
    return {
        "offense": {"goals_per_game": round(float(r["scored"]), 2)},
        "defense": defense,
    }


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


def _nhl_name(entry):
    return f"{entry['firstName']['default']} {entry['lastName']['default']}"


def build_nhl_standings():
    """Real, current NHL standings straight from the league's own standings
    endpoint -- unlike NFL/MLB there's no win-loss aggregation to do
    ourselves, the API already computes division/conference rank, points,
    and record. Confirmed live: /now always reflects the most relevant
    current standings (mid-season while games are being played, or the just-
    finished season's final table once it's over), so no separate
    off-season fallback is needed the way NFL's build needed one."""
    resp = requests.get("https://api-web.nhle.com/v1/standings/now", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    by_division = {}
    for t in data["standings"]:
        streak = None
        if t.get("streakCode") and t.get("streakCount"):
            streak = f"{t['streakCode']}{t['streakCount']}"
        row = {
            "team": normalize_team(t["teamAbbrev"]["default"]),
            "division": t["divisionName"], "conference": t["conferenceName"],
            "rank": t["divisionSequence"],
            "games_played": t["gamesPlayed"], "wins": t["wins"], "losses": t["losses"],
            "ot_losses": t["otLosses"], "points": t["points"],
            "point_pct": round(t["pointPctg"], 3), "streak": streak,
        }
        by_division.setdefault(row["division"], []).append(row)
    for div_rows in by_division.values():
        div_rows.sort(key=lambda r: r["rank"])

    season_id = data["standings"][0]["seasonId"] if data["standings"] else None
    return {"season": season_id, "as_of": data.get("standingsDateTimeUtc"), "standings": by_division}


def build_nhl_stat_leaders(top_n=5):
    """Real current-season stat leaders straight from the league's own
    leaders endpoints -- no aggregation needed, and no separate data source
    from what build_nhl_standings already hits (same api-web.nhle.com base)."""
    skater_resp = requests.get("https://api-web.nhle.com/v1/skater-stats-leaders/current", timeout=15)
    skater_resp.raise_for_status()
    skaters = skater_resp.json()

    goalie_resp = requests.get("https://api-web.nhle.com/v1/goalie-stats-leaders/current", timeout=15)
    goalie_resp.raise_for_status()
    goalies = goalie_resp.json()

    def top_leaders(entries, label, value_round=None):
        top = entries[:top_n]
        return {"stat": label, "leaders": [
            {
                "player": _nhl_name(e), "team": normalize_team(e["teamAbbrev"]),
                "value": round(e["value"], value_round) if value_round is not None else e["value"],
            }
            for e in top
        ]}

    return {
        "skaters": [
            top_leaders(skaters.get("points", []), "Points"),
            top_leaders(skaters.get("goals", []), "Goals"),
            top_leaders(skaters.get("assists", []), "Assists"),
        ],
        "goalies": [
            top_leaders(goalies.get("wins", []), "Wins"),
            top_leaders(goalies.get("savePctg", []), "Save %", value_round=3),
        ],
    }


def _nhl_remaining_games(today):
    """Real, still-to-be-played regular-season games from today through the
    end of the current season -- data/nhl/team_games.parquet only has
    completed games (pulled from the same schedule endpoint but filtered to
    finished game states), so future games need their own live pull here,
    same _fetch_week/window-stepping pattern pull_games.py uses for
    historical seasons, just walked forward instead of over a fixed
    already-known range. Deliberately does NOT rely on pull_games.py's
    SEASON_WINDOWS for an end date -- that list only gets a new entry added
    by hand once a season's real dates are known, so it can genuinely lag
    behind "today" (confirmed: it does right now). Steps forward up to 30
    weeks (a real NHL regular season is ~26) and stops early once a whole
    week comes back with zero real games -- the schedule endpoint returns an
    empty week for real gaps between seasons, same signal
    detect_target_date already relies on elsewhere in this file."""
    d = pd.Timestamp(today)
    games = []
    seasons_seen = set()
    seen_ids = set()
    empty_weeks_seen = 0
    for _ in range(30):
        data = _fetch_week(d.strftime("%Y-%m-%d"))
        week_had_games = False
        for week in data.get("gameWeek", []):
            for g in week.get("games", []):
                if g.get("gameType") != 2:
                    continue
                week_had_games = True
                # NHL's own season id, e.g. 20262027 -> this function's
                # caller needs the start-year (2026) to match
                # team_games.parquet's convention.
                if g.get("season"):
                    seasons_seen.add(int(str(g["season"])[:4]))
                if g.get("gameState") in ("OFF", "FINAL") or g["id"] in seen_ids:
                    continue
                seen_ids.add(g["id"])
                games.append({
                    "home_team": normalize_team(g["homeTeam"]["abbrev"]),
                    "away_team": normalize_team(g["awayTeam"]["abbrev"]),
                })
        empty_weeks_seen = 0 if week_had_games else empty_weeks_seen + 1
        if empty_weeks_seen >= 2 and games:
            break  # two straight empty weeks after real games means the season's over
        d += pd.Timedelta(days=7)
    season = max(seasons_seen) if seasons_seen else None
    return pd.DataFrame(games), season


def build_nhl_title_odds():
    """Division title + playoff berth odds, from Monte Carlo simulating the
    real remaining schedule from each team's current real Elo rating --
    pipeline/common/season_sim.py, validated in
    pipeline/nhl/backtest_season_sim.py against 5 real completed NHL seasons
    (beats a naive win-rate-extrapolation baseline in 9/10 backtested
    snapshots; pooled division-title Brier 0.0740 vs. a naive equal-
    probability baseline's 0.1094). Division-title rates are shrunk toward
    the field (shrinkage=0.85, NHL's own backtested value); playoff-berth
    rates ship as the simulator's raw output (not separately calibration-
    checked). Real NHL playoff format: top 3 in each of 2 divisions per
    conference guaranteed (top_n_per_division=3), plus 2 more wildcards per
    conference (berths_per_conference=8)."""
    from pipeline.common.season_sim import simulate_remaining_wins, division_title_rates, playoff_berth_rates, shrink_toward_field

    with open(ROOT / "notebooks_out" / "nhl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    played = load_games()  # all real completed games, every season on record
    _, ratings = run_elo(played, k=elo_params["k"], home_adv=elo_params["home_adv"],
                          scale=elo_params["scale"], season_regression=elo_params["season_regression"],
                          return_ratings=True)

    today = datetime.date.today()
    remaining, target_season = _nhl_remaining_games(today)
    if target_season is None:
        target_season = int(played["season"].max())  # fallback: no real upcoming games found at all

    # current_wins has to come from the SAME season as `remaining` -- using
    # played["season"].max() here would silently be last season's full,
    # final win totals (real bug caught in testing: during the real
    # off-season, played's max season is the one that just ended, while
    # remaining is already next season's schedule; adding a finished
    # season's ~50+ real wins on top of a few simulated new-season wins
    # made the just-finished top teams show ~100% title odds for a season
    # they haven't played a single real game of yet).
    current_wins = {}
    for _, r in played[played["season"] == target_season].iterrows():
        current_wins.setdefault(r["home_team"], 0)
        current_wins.setdefault(r["away_team"], 0)
        if r["home_win"] == 1.0:
            current_wins[r["home_team"]] += 1
        else:
            current_wins[r["away_team"]] += 1

    resp = requests.get("https://api-web.nhle.com/v1/standings/now", timeout=15)
    resp.raise_for_status()
    standings_data = resp.json()
    team_divisions = {normalize_team(t["teamAbbrev"]["default"]): t["divisionName"] for t in standings_data["standings"]}
    team_conferences = {normalize_team(t["teamAbbrev"]["default"]): t["conferenceName"] for t in standings_data["standings"]}
    ratings = {t: r for t, r in ratings.items() if t in team_divisions}

    teams, sim_wins = simulate_remaining_wins(remaining, ratings, elo_params["home_adv"], elo_params["scale"], n_sims=5000)
    base_wins = np.array([current_wins.get(t, 0) for t in teams])
    final_wins = base_wins[None, :] + sim_wins

    div_rates = division_title_rates(teams, final_wins, team_divisions)
    for team_rates in div_rates.values():
        for t in team_rates:
            team_rates[t] = round(shrink_toward_field(team_rates[t], 8, 0.85), 4)

    playoff_rates = playoff_berth_rates(teams, final_wins, team_conferences, berths_per_conference=8,
                                        division_winners_guaranteed=team_divisions, top_n_per_division=3)

    return {"division_title_pct": div_rates, "playoff_pct": playoff_rates}


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
            "awayTeamStats": team_stats_for_dropdown(scoring_rates, g["away_team"]),
            "homeTeamStats": team_stats_for_dropdown(scoring_rates, g["home_team"]),
        }
        by_date.setdefault(g["target_date"], []).append(out_game)

    for d, day_games in by_date.items():
        days_out[d] = build_day_payload(d, day_games)

    _write_payload(dates, anchor_iso, days_out, elo_params)


def _write_payload(dates, today_iso, days_out, elo_params=None):
    print("Building NHL standings + stat leaders...")
    nhl_standings = build_nhl_standings()
    nhl_title_odds = build_nhl_title_odds()
    record_title_odds("nhl", nhl_title_odds, snapshot_date=today_iso, season=nhl_standings.get("season"))
    payload = {
        "week_start": dates[0], "week_end": dates[-1], "today": today_iso,
        "elo_params": elo_params,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "days": days_out,
        "season_info": {"standings": nhl_standings, "stat_leaders": build_nhl_stat_leaders(),
                        "title_odds": nhl_title_odds},
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "dashboard_current_slate.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    total_games = sum(len(day["games"]) for day in days_out.values())
    print(f"\nWrote {out_path} -- {len(days_out)} days, {total_games} total games")


if __name__ == "__main__":
    main()

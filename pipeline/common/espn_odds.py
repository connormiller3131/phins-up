"""Free, unauthenticated fallback source for real game-line odds
(moneyline/run-line-or-spread/total), via ESPN's public site API. Used only
to fill in whatever attach_market_odds's primary DraftKings-via-The-Odds-API
call couldn't price -- confirmed real case: every single MLB game came back
with a null moneyline on a refresh, which is what happens when that bulk
call throws entirely (almost certainly a credit-quota exhaustion, since MLB
also spends per-event player-prop credits twice a day), not a per-game gap.
ESPN's scoreboard/summary endpoints have no visible request quota, so this
removes that whole failure mode for game lines specifically. It can't
replace The Odds API outright: attach_featured_prop_odds still needs the
event_id that call returns to fetch player props from that same account.
"""
import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

SPORT_PATHS = {
    "baseball_mlb": "baseball/mlb",
    "americanfootball_nfl": "football/nfl",
}


def get_scoreboard_events(sport_key, date_iso=None):
    """Real games (if any) for one date. Team names come back as ESPN's
    `displayName` ("Philadelphia Phillies"), the same full-name format
    team_map.py's full_name_to_statcast already knows how to translate --
    no separate ESPN-specific abbreviation table needed."""
    params = {"dates": date_iso.replace("-", "")} if date_iso else {}
    resp = requests.get(f"{BASE_URL}/{SPORT_PATHS[sport_key]}/scoreboard", params=params, timeout=15)
    resp.raise_for_status()
    out = []
    for e in resp.json().get("events", []):
        comp = e["competitions"][0]
        home = next((c for c in comp["competitors"] if c["homeAway"] == "home"), None)
        away = next((c for c in comp["competitors"] if c["homeAway"] == "away"), None)
        if not home or not away:
            continue
        out.append({
            "event_id": e["id"],
            "home_name": home["team"]["displayName"],
            "away_name": away["team"]["displayName"],
        })
    return out


def get_event_odds(sport_key, event_id):
    """Real moneyline/spread/total for one event, preferring the DraftKings
    entry in ESPN's `pickcenter` (matches the bookmaker The Odds API pulls
    everywhere else in this repo) and falling back to whichever provider is
    listed first if DK isn't posted there yet. None if ESPN has no line for
    this event yet (same meaning as an unposted line anywhere else)."""
    resp = requests.get(f"{BASE_URL}/{SPORT_PATHS[sport_key]}/summary", params={"event": event_id}, timeout=15)
    resp.raise_for_status()
    pickcenter = resp.json().get("pickcenter", [])
    if not pickcenter:
        return None
    entry = next((p for p in pickcenter if p.get("provider", {}).get("name") == "DraftKings"), pickcenter[0])
    home_ml = entry.get("homeTeamOdds", {}).get("moneyLine")
    away_ml = entry.get("awayTeamOdds", {}).get("moneyLine")
    if home_ml is None or away_ml is None:
        return None
    return {
        "mlHome": home_ml, "mlAway": away_ml,
        "run_line_home": entry.get("spread"), "total_line": entry.get("overUnder"),
    }

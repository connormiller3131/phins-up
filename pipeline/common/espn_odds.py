"""Free, unauthenticated wrapper around ESPN's public site API, covering two
unrelated gaps in this pipeline:

- Fallback game-line odds (moneyline/run-line-or-spread/total). Used only to
  fill in whatever attach_market_odds's primary DraftKings-via-The-Odds-API
  call couldn't price -- confirmed real case: every single MLB game came
  back with a null moneyline on a refresh, which is what happens when that
  bulk call throws entirely (almost certainly a credit-quota exhaustion,
  since MLB also spends per-event player-prop credits twice a day), not a
  per-game gap. It can't replace The Odds API outright: attach_featured_prop_odds
  still needs the event_id that call returns to fetch player props from
  that same account.

- Confirmed starting lineups, with real batting order, once each team has
  posted theirs -- ESPN's per-event summary carries this well before first
  pitch in practice (confirmed against a still-STATUS_SCHEDULED game ~1.5
  hours out that already had all 9 real starters posted). This replaces
  this pipeline's previous "no confirmed lineup source" gap, where the
  starting 9 batters were only ever a proxy (top 9 by trailing PA volume).

Both endpoints have no visible request quota."""
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
            "commence_time": e.get("date"),
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


def get_confirmed_lineup(sport_key, event_id):
    """The real starting 9-batter lineup + batting order for one event, once
    posted. Returns {'home': [...], 'away': [...]} (each a list of {'name',
    'bat_order'} in batting order) or None if ESPN doesn't have it yet.
    ESPN's own athlete ids are a different id space from this repo's MLBAM
    player_id (Statcast/pybaseball convention) -- callers need to resolve
    `name` against their own player lookup, not treat the id as usable."""
    resp = requests.get(f"{BASE_URL}/{SPORT_PATHS[sport_key]}/summary", params={"event": event_id}, timeout=15)
    resp.raise_for_status()
    out = {}
    for side in resp.json().get("rosters", []):
        home_away = side.get("homeAway")
        lineup = [
            {"name": p["athlete"]["fullName"], "bat_order": p["batOrder"]}
            for p in side.get("roster", [])
            if p.get("starter") and p.get("batOrder") is not None
        ]
        if lineup:
            out[home_away] = sorted(lineup, key=lambda x: x["bat_order"])
    return out or None

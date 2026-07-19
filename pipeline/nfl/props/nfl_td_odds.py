"""Real DraftKings odds via The Odds API, for two things:
1. "Current" game moneylines -- a cheap bulk call, attached to every week
   the book has actually posted a line for (may or may not extend past what
   nflverse's own opening-line data covers).
2. Real anytime-TD player-prop odds -- an expensive per-event call, so this
   is restricted to the CURRENT week only. Books don't post player props
   until close to kickoff anyway, so pulling this for future weeks would
   just spend credits on empty responses.
Fails soft at every stage: if ODDS_API_KEY isn't set, or DK hasn't posted a
line yet, the affected fields are simply left unattached rather than
breaking the run."""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.common.odds_api import get_game_odds, get_event_player_props

SPORT_KEY = "americanfootball_nfl"


def _implied_prob(american_odds):
    return (-american_odds) / (-american_odds + 100) if american_odds < 0 else 100 / (american_odds + 100)


def fetch_current_week_odds_map(names):
    """One bulk call (cheap, a few credits for the whole slate) for DK's
    current h2h game lines. Returns {(away_full_name, home_full_name):
    {"event_id":..., "mlAway":..., "mlHome":...}}, empty dict on any failure."""
    try:
        events = get_game_odds(SPORT_KEY, markets="h2h")
    except Exception as e:
        print(f"[nfl_td_odds] bulk odds call failed, skipping current-line + TD odds: {e}")
        return {}

    out = {}
    for ev in events:
        dk = next((b for b in ev.get("bookmakers", []) if b.get("key") == "draftkings"), None)
        ml_away = ml_home = None
        if dk:
            for mkt in dk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for outcome in mkt.get("outcomes", []):
                    if outcome["name"] == ev["away_team"]:
                        ml_away = int(outcome["price"])
                    elif outcome["name"] == ev["home_team"]:
                        ml_home = int(outcome["price"])
        out[(ev["away_team"], ev["home_team"])] = {"event_id": ev["id"], "mlAway": ml_away, "mlHome": ml_home}
    return out


def attach_current_lines(games_out, names, odds_map):
    """Cheap -- safe to call for every week. Adds current_mlAway/current_mlHome
    /current_market_home_prob where DraftKings has a live line posted."""
    for g in games_out:
        key = (names.get(g["awayAbbr"], g["awayAbbr"]), names.get(g["homeAbbr"], g["homeAbbr"]))
        match = odds_map.get(key)
        if not match or match["mlAway"] is None or match["mlHome"] is None:
            continue
        pa, ph = _implied_prob(match["mlAway"]), _implied_prob(match["mlHome"])
        g["current_mlAway"] = match["mlAway"]
        g["current_mlHome"] = match["mlHome"]
        g["current_market_home_prob"] = round(ph / (pa + ph), 4)


def attach_td_odds(games_out, names, odds_map):
    """Expensive -- only call this for the current week. For each game's
    Anytime TD prop entries, adds a "dk_odds" field where DraftKings has
    posted a line for that player, matched by name."""
    for g in games_out:
        key = (names.get(g["awayAbbr"], g["awayAbbr"]), names.get(g["homeAbbr"], g["homeAbbr"]))
        match = odds_map.get(key)
        if not match:
            continue

        td_props = [p for p in g["props"] if p["market"] == "Anytime TD"]
        if not td_props:
            continue

        try:
            event_odds = get_event_player_props(SPORT_KEY, match["event_id"], markets="player_anytime_td")
        except Exception as e:
            print(f"[nfl_td_odds] event odds call failed for {g['awayAbbr']}@{g['homeAbbr']}: {e}")
            continue

        dk_by_player = {}
        for bm in event_odds.get("bookmakers", []):
            if bm.get("key") != "draftkings":
                continue
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "player_anytime_td":
                    continue
                for outcome in mkt.get("outcomes", []):
                    player = outcome.get("description") or outcome.get("name")
                    if player:
                        dk_by_player[player] = int(outcome["price"])

        for p in td_props:
            if p["player"] in dk_by_player:
                p["dk_odds"] = dk_by_player[p["player"]]
                p["dk_implied_prob"] = round(_implied_prob(dk_by_player[p["player"]]), 4)

"""Real DraftKings anytime-TD odds via The Odds API, attached to the current
week's games only (not the full season) -- per-event player-prop pulls cost
API credits per call, and books don't post player props until close to
kickoff anyway, so pulling this for future weeks would just waste credits on
empty responses. Fails soft: if ODDS_API_KEY isn't set, or DK hasn't posted
TD props for a game yet, that game's props simply get no odds attached."""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.common.odds_api import get_game_odds, get_event_player_props

SPORT_KEY = "americanfootball_nfl"


def _event_ids_by_team_pair(names):
    """Bulk call (cheap) to find each upcoming game's Odds API event ID,
    keyed by (away full name, home full name)."""
    try:
        events = get_game_odds(SPORT_KEY, markets="h2h")
    except Exception as e:
        print(f"[nfl_td_odds] bulk odds call failed, skipping TD odds: {e}")
        return {}
    return {(ev["away_team"], ev["home_team"]): ev["id"] for ev in events}


def attach_td_odds(games_out, names):
    """Mutates games_out in place: for each game's Anytime TD prop entries,
    adds a "dk_odds" field (American odds int) where DraftKings has posted a
    line for that player, matched by name. Games/players without a posted
    line are left untouched."""
    by_pair = _event_ids_by_team_pair(names)
    if not by_pair:
        return games_out

    for g in games_out:
        away_full = names.get(g["awayAbbr"], g["awayAbbr"])
        home_full = names.get(g["homeAbbr"], g["homeAbbr"])
        event_id = by_pair.get((away_full, home_full))
        if not event_id:
            continue

        td_props = [p for p in g["props"] if p["market"] == "Anytime TD"]
        if not td_props:
            continue

        try:
            event_odds = get_event_player_props(SPORT_KEY, event_id, markets="player_anytime_td")
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

    return games_out

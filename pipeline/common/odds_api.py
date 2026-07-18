"""Thin client for The Odds API (the-odds-api.com), used for real DraftKings
game-line odds. Reads the API key from the ODDS_API_KEY environment variable
(never hardcode it -- this repo is public) so it works the same way locally
and as a GitHub Actions secret."""
import os
import requests

BASE_URL = "https://api.the-odds-api.com/v4"


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY environment variable not set. Get a free key at "
            "https://the-odds-api.com and set it before running odds-dependent scripts."
        )
    return key


def get_game_odds(sport_key: str, bookmaker: str = "draftkings", markets: str = "h2h,spreads,totals"):
    """Bulk game-lines pull for an entire sport's upcoming slate. Cheap: costs
    (# markets) credits total for ALL games in one call, regardless of slate size."""
    resp = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds",
        params={
            "apiKey": _api_key(),
            "regions": "us",
            "markets": markets,
            "bookmakers": bookmaker,
            "oddsFormat": "american",
        },
        timeout=30,
    )
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"[odds_api] credits used this call, remaining: {remaining} (used so far this period: {used})")
    return resp.json()


def get_event_player_props(sport_key: str, event_id: str, markets: str, bookmaker: str = "draftkings"):
    """Per-event player-prop pull. Costs (# markets) credits PER CALL -- only
    use this for a deliberately limited set of games, not the full slate,
    unless on a paid plan with headroom."""
    resp = requests.get(
        f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
        params={
            "apiKey": _api_key(),
            "regions": "us",
            "markets": markets,
            "bookmakers": bookmaker,
            "oddsFormat": "american",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

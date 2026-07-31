"""Thin client for The Odds API (the-odds-api.com), used for real DraftKings
game-line odds. Reads the API key from the ODDS_API_KEY environment variable
(never hardcode it -- this repo is public) so it works the same way locally
and as a GitHub Actions secret."""
import os
import json
import pathlib
import datetime
import requests
import pandas as pd

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
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"[odds_api] event props call for {event_id}, credits remaining: {remaining} (used so far this period: {used})")
    return resp.json()


# ---------------------------------------------------------------------------
# Disk-backed caching so a game's odds get fetched once and reused, instead
# of every refresh.yml run (twice daily) re-pulling the same not-yet-played
# game's props for however many days it sits in "upcoming" -- confirmed real
# waste: a game generated up to a week ahead could get re-fetched 10+ times
# before it's even played. Cache files are plain JSON, persisted across CI
# runs via the same GitHub Actions cache pattern already used for historical
# pull data (see refresh.yml) -- this is ephemeral, regenerable data, not
# worth git-committing the way NHL's goalie pull was.
# ---------------------------------------------------------------------------

def _load_cache(cache_path: pathlib.Path):
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: pathlib.Path, cache: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _ttl_hours(commence_time):
    """Longer cache lifetime for a game still days out (nothing meaningful
    changes hour to hour that far ahead); shorter as first pitch/kickoff
    gets close, since that's when lines move fastest and freshness matters
    most. Falls back to a flat middle-ground TTL if commence_time isn't
    known (shouldn't normally happen -- every real event has one)."""
    if not commence_time:
        return 12
    try:
        hours_until = (pd.Timestamp(commence_time) - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600
    except (ValueError, TypeError):
        return 12
    if hours_until > 48:
        return 24
    if hours_until > 24:
        return 12
    return 4


def cached_event_player_props(cache_path: pathlib.Path, sport_key: str, event_id: str, markets: str,
                               commence_time=None, bookmaker: str = "draftkings"):
    """Same real data as get_event_player_props, but skips the network call
    entirely if this exact event+markets combination was already fetched
    recently enough (see _ttl_hours) -- this is the single most expensive
    call pattern in the pipeline (costs credits per market PER CALL), so
    it's the one most worth not repeating for no reason."""
    cache = _load_cache(cache_path)
    key = f"{event_id}:{markets}"
    entry = cache.get(key)
    now = datetime.datetime.now(datetime.timezone.utc)
    ttl = _ttl_hours(commence_time)
    if entry:
        fetched_at = datetime.datetime.fromisoformat(entry["fetched_at"])
        if (now - fetched_at) < datetime.timedelta(hours=ttl):
            print(f"[odds_api] cache hit for event {event_id} (fetched {fetched_at.isoformat()}, ttl {ttl}h) -- no call made")
            return entry["data"]
    data = get_event_player_props(sport_key, event_id, markets, bookmaker)
    cache[key] = {"fetched_at": now.isoformat(), "data": data}
    _save_cache(cache_path, cache)
    return data


def cached_game_odds(cache_path: pathlib.Path, sport_key: str, bookmaker: str = "draftkings",
                      markets: str = "h2h,spreads,totals", ttl_hours: float = 3):
    """Same idea as cached_event_player_props but for the bulk endpoint --
    already cheap per call, but still no reason to hit it more than once
    within a short window if a run gets triggered twice in quick
    succession (e.g. a manual run alongside the scheduled one)."""
    cache = _load_cache(cache_path)
    key = f"bulk:{markets}"
    entry = cache.get(key)
    now = datetime.datetime.now(datetime.timezone.utc)
    if entry:
        fetched_at = datetime.datetime.fromisoformat(entry["fetched_at"])
        if (now - fetched_at) < datetime.timedelta(hours=ttl_hours):
            print(f"[odds_api] cache hit for bulk {sport_key} odds (fetched {fetched_at.isoformat()}) -- no call made")
            return entry["data"]
    data = get_game_odds(sport_key, bookmaker, markets)
    cache[key] = {"fetched_at": now.isoformat(), "data": data}
    _save_cache(cache_path, cache)
    return data

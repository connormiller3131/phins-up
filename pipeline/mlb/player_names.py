"""MLBAM player ID -> display name lookup, via pybaseball's Chadwick register,
with a real-MLB-API fallback for anyone Chadwick doesn't have yet.

Not disk-cached across runs -- a permanent on-disk cache here previously went
stale (built once, then silently missing every player who debuted since),
which meant roughly half of all current batters had no display name at all
by the time this was caught, and that NaN crashed the pipeline downstream
(see current_state.py's _current_trailing). The real fetch itself is a
single ~26k-row register, ~0.4s -- there's no cost to justify caching it, so
it's just refetched fresh every run and memoized in-process only (so a
single run's many callers don't refetch it repeatedly).

Even fresh, Chadwick's register is missing ~1,300 of the batter IDs and
~600 of the pitcher IDs actually appearing in current Statcast data --
confirmed these are real, current players (e.g. Cristhian Vaquero, Brennen
Davis) that Chadwick's community-maintained register just hasn't caught up
to, not a caching or ID-format issue. fetch_names_for_ids() fills exactly
those gaps from MLB's own Stats API (the same MLBAM ID space, so no
mapping issues), in bulk (100 IDs/request) rather than one call per player,
memoized so the many _current_trailing calls across one pipeline run don't
refetch the same IDs repeatedly."""
import warnings
import requests
import pandas as pd
import pybaseball as pb

warnings.filterwarnings("ignore")

_cache = None
_api_name_cache = {}


def get_name_lookup():
    global _cache
    if _cache is not None:
        return _cache

    reg = pb.chadwick_register()
    reg = reg[reg["key_mlbam"].notna()].copy()
    reg["player_id"] = reg["key_mlbam"].astype(int)
    reg["player_display_name"] = reg["name_first"].fillna("") + " " + reg["name_last"].fillna("")
    reg["player_display_name"] = reg["player_display_name"].str.strip()
    out = reg[["player_id", "player_display_name"]].drop_duplicates("player_id")

    _cache = out
    return out


def fetch_names_for_ids(missing_ids):
    """Real names for player_ids Chadwick's register doesn't have, fetched
    from MLB's own Stats API (bulk, 100 IDs per request). Returns a dict of
    whatever it found; any ID MLB's API itself doesn't recognize either is
    just omitted (caller falls back to a placeholder for those)."""
    to_fetch = [int(pid) for pid in missing_ids if int(pid) not in _api_name_cache]
    for i in range(0, len(to_fetch), 100):
        batch = to_fetch[i:i + 100]
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/people",
                params={"personIds": ",".join(str(pid) for pid in batch)},
                timeout=15,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                _api_name_cache[int(person["id"])] = person["fullName"]
        except Exception as e:
            print(f"  name fallback fetch failed for {len(batch)} ids: {e}")

    return {pid: _api_name_cache[int(pid)] for pid in missing_ids if int(pid) in _api_name_cache}


if __name__ == "__main__":
    lut = get_name_lookup()
    print(lut.shape)
    print(lut.head())

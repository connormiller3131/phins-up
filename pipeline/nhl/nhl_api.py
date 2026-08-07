"""Polite, retrying client for api-web.nhle.com.

A single NHL generator run can fire well over a hundred requests at this
host: detect_target_date walks forward a week at a time looking for the next
scheduled game, _nhl_remaining_games walks forward again for the rest of the
season, then the slate, standings and stat-leader calls follow. In the
off-season those walks are at their longest, because the next game can be
two months out.

Fired back-to-back with no pacing, that reliably trips the API's rate limit,
and a bare requests.get raises straight through and kills the whole refresh
-- confirmed on real runs: three scheduled refreshes failed on
"429 Client Error: Too Many Requests", one on /v1/schedule and one on
/v1/skater-stats-leaders, taking the already-generated NFL and MLB output
down with them.

So: one shared session (connection reuse), a small gap between calls, and
exponential backoff that honours Retry-After when the server sends it.
"""
import time
import requests

BASE = "https://api-web.nhle.com/v1"

# Minimum gap between consecutive calls to this host. Small enough not to
# meaningfully slow a run (a 100-call run costs ~12s), large enough to stop
# a tight schedule-walk loop from hammering the API.
MIN_INTERVAL = 0.12
MAX_ATTEMPTS = 5
BACKOFF_BASE = 1.6

_session = None
_last_call = 0.0


def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "phinsup.net projections (contact: phinsupdotnet2@gmail.com)"})
    return _session


def get_json(url, timeout=15):
    """GET returning parsed JSON, retrying on rate limits and transient
    server errors. Raises on a genuine client error (a 404 means the URL is
    wrong and retrying will not help) and after the final attempt."""
    global _last_call
    last_exc = None
    for attempt in range(MAX_ATTEMPTS):
        gap = time.monotonic() - _last_call
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        try:
            resp = _get_session().get(url, timeout=timeout)
            _last_call = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                # Retry-After is authoritative when present; otherwise back
                # off exponentially rather than guessing a flat delay.
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else BACKOFF_BASE ** attempt
                except ValueError:
                    wait = BACKOFF_BASE ** attempt
                if attempt < MAX_ATTEMPTS - 1:
                    print(f"  [nhl_api] {resp.status_code} from {url.rsplit('/', 2)[-2:]}, "
                          f"retrying in {wait:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_exc = e
            _last_call = time.monotonic()
            status = getattr(getattr(e, "response", None), "status_code", None)
            # 4xx other than 429 is a real, permanent error -- fail fast.
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt < MAX_ATTEMPTS - 1:
                wait = BACKOFF_BASE ** attempt
                print(f"  [nhl_api] request failed ({e.__class__.__name__}), "
                      f"retrying in {wait:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
                time.sleep(wait)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"NHL API request failed after {MAX_ATTEMPTS} attempts: {url}")

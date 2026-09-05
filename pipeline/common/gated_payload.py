"""Split each sport's dashboard payload into a free half and a gated half.

The site ships as one HTML file with every projection inlined, and that file
is committed to a PUBLIC repo which GitHub Pages also serves in its own
right (connormiller3131.github.io/phins-up). Verified directly: the public
copy is 7.96 MB and contains all 3003 Anytime TD props. So anything left
inlined is free to the world no matter what the page or the Worker does
about it -- gating in the browser would be decoration, and the data would
additionally live in git history forever.

The free half stays inlined and committed. The gated half is written to
data/ (already git-ignored) and uploaded to Cloudflare KV by the refresh
workflow, so it reaches a browser only through site-worker.js, which is the
single place Phase 1 adds an entitlement check.

Measured on a real build, props are 86% of an NFL game's payload (11.5 KB
vs 1.9 KB) -- so this also drops the page most of its weight, which is why
it is worth doing on its own merits even before anything is charged for.

Splitting is by POP, not copy: the gated keys are lifted out of the loaded
structure and collected, leaving the same object as the free version. There
is never a second copy of an 8 MB payload in memory.
"""

# Keys lifted off an individual game object. hr_combo is the MLB "either/or"
# homer special and featured_props the real-DK-priced subset; both are
# derived from the same prop models as props itself, so they gate together.
# featured_props is not currently emitted by the MLB pipeline, but
# mlbPropsTable still reads it, so it is listed here rather than left as a
# hole that silently opens if it comes back.
NFL_GATED_GAME_KEYS = ("props",)
MLB_GATED_GAME_KEYS = ("props", "hr_combo", "featured_props")


def _lift(game, keys):
    """Pop every gated key present on one game. Returns None when the game
    carries none of them, so an already-played or propless game doesn't
    produce an empty entry that the client would then try to merge."""
    lifted = {k: game.pop(k) for k in keys if k in game}
    return lifted or None


def nfl_game_key(week, game):
    """Stable identity for one NFL game. Matched up client-side rather than
    relying on array position: the free half is baked into a committed page
    while the gated half is uploaded separately, so the two can be built
    minutes apart and must not depend on their orderings staying aligned."""
    return f"{week}|{game.get('awayAbbr')}|{game.get('homeAbbr')}"


def mlb_game_key(game):
    """MLB carries MLB's own gamePk, which is already unique per game and
    survives doubleheaders (two games, same teams, same date, distinct
    gamePk) where an abbreviation pair would collide."""
    return str(game.get("gamePk"))


def split_nfl(data):
    """Mutates data into its free form; returns the gated map."""
    gated = {}
    for week, wk in (data.get("weeks") or {}).items():
        for game in wk.get("games") or []:
            lifted = _lift(game, NFL_GATED_GAME_KEYS)
            if lifted:
                gated[nfl_game_key(week, game)] = lifted
    return gated


def split_mlb(data):
    """Mutates data into its free form; returns the gated map."""
    gated = {}
    for day in (data.get("days") or {}).values():
        for game in day.get("games") or []:
            lifted = _lift(game, MLB_GATED_GAME_KEYS)
            if lifted:
                gated[mlb_game_key(game)] = lifted
    return gated


def build_gated_payload(nfl_data, mlb_data):
    """Split both sports and wrap the gated halves with the build stamp the
    client checks before merging. NHL has no player props at all (its tab is
    win-probability only), so it is free in its entirety and absent here.

    build is NFL's generated_at because that is the run that produces the
    committed page. If a stale KV value ever gets served against a newer
    page, the mismatch is visible to the client rather than silently
    merging last night's props onto today's games."""
    gated = {
        "build": nfl_data.get("generated_at"),
        "nfl": split_nfl(nfl_data),
        "mlb": split_mlb(mlb_data),
    }
    return gated

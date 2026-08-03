"""Append-only history of each refresh's computed season odds.

The season simulation recomputes division-title and playoff-berth
probabilities from scratch on every refresh and the payload only ever holds
the current values, so before this existed there was no record of what any
team's odds were yesterday. That makes "odds over time" charts (the shape
FanGraphs' playoff-odds graphs use) impossible to build retroactively --
history only exists if something starts writing it down.

Output lands in docs/results/, which is already git-committed by refresh.yml
and served as a static file, so snapshots survive CI's fresh-checkout-every-
run model the same way the graded prediction snapshots next to them do.

Keyed by date, not by run: refresh runs twice daily, and the later run
overwrites the earlier one's entry for that date rather than adding a second
point. A daily granularity is all these numbers meaningfully move at, and it
keeps one refresh being manually re-triggered from putting a visible kink in
the chart.
"""
import json
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "docs" / "results"


def history_path(sport: str) -> pathlib.Path:
    return HISTORY_DIR / f"odds_history_{sport}.json"


def _flatten(title_odds: dict) -> dict:
    """{"division_title_pct": {div: {team: p}}, "playoff_pct": {team: p}}
    -> {team: {"div": p, "po": p}}. Division nesting is only there so the
    standings table can group rows; for a per-team time series it just adds
    a lookup step, so it gets flattened on the way in."""
    teams = {}
    for div_rates in (title_odds.get("division_title_pct") or {}).values():
        for team, p in (div_rates or {}).items():
            if p is not None:
                teams.setdefault(team, {})["div"] = round(float(p), 4)
    for team, p in (title_odds.get("playoff_pct") or {}).items():
        if p is not None:
            teams.setdefault(team, {})["po"] = round(float(p), 4)
    return teams


def _load(path: pathlib.Path, sport: str) -> dict:
    if not path.exists():
        return {"sport": sport, "snapshots": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("snapshots"), dict):
            return data
        raise ValueError("missing 'snapshots' object")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        # Never silently discard real accumulated history: move the
        # unreadable file aside (so it can still be inspected/recovered by
        # hand) and start a fresh one, rather than either crashing the whole
        # generator or overwriting it in place.
        backup = path.with_suffix(".corrupt.json")
        try:
            path.replace(backup)
            print(f"  [odds_history] {path.name} unreadable ({e}); moved to {backup.name}, starting fresh")
        except OSError:
            print(f"  [odds_history] {path.name} unreadable ({e}) and could not be moved aside; skipping record")
            return None
        return {"sport": sport, "snapshots": {}}


def record_title_odds(sport: str, title_odds: dict, snapshot_date=None, season=None) -> None:
    """Record one dated snapshot. Best-effort by design: a failure here must
    never take down a refresh whose real job is the daily slate, so problems
    print and return instead of raising."""
    if not title_odds:
        print(f"  [odds_history] no title odds for {sport}, nothing recorded")
        return

    teams = _flatten(title_odds)
    if not teams:
        print(f"  [odds_history] title odds for {sport} had no per-team values, nothing recorded")
        return

    date_key = snapshot_date or datetime.date.today().isoformat()
    path = history_path(sport)
    data = _load(path, sport)
    if data is None:
        return

    entry = {"teams": teams}
    if season is not None:
        entry["season"] = season
    is_new = date_key not in data["snapshots"]
    data["snapshots"][date_key] = entry
    data["sport"] = sport
    data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=True)
    except OSError as e:
        print(f"  [odds_history] could not write {path.name}: {e}")
        return

    verb = "recorded" if is_new else "updated"
    print(f"  [odds_history] {verb} {sport} odds for {date_key} "
          f"({len(teams)} teams, {len(data['snapshots'])} day(s) of history)")

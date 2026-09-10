"""Grades pregame prediction snapshots (docs/results/nfl_*.json, written by
generate_current_week.py the first time each game's props are generated)
against real results, once those games have actually been played. Run this
after pull_data.py refreshes schedules.parquet/player_stats.parquet and
before generate_current_week.py, so newly-completed games get graded before
the next batch of pregame snapshots is written.

Snapshots are never overwritten once graded=true, so this only touches
games that just finished since the last run."""
import sys
import pathlib
import json
import pandas as pd
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "nfl"
RESULTS_DIR = ROOT / "docs" / "results"
# Where generate_current_week.py parks the frozen pregame props. Git-ignored,
# and restored from Cloudflare KV by refresh.yml before this runs, because
# docs/results/ is committed and served publicly -- a pregame prop list there
# is the paid product given away.
PENDING_PROPS_PATH = ROOT / "data" / "pending_props.json"


def load_pending_props():
    if PENDING_PROPS_PATH.exists():
        with open(PENDING_PROPS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def drop_pending_props(names):
    """Once a game is graded its props live in the public snapshot as
    props_graded, so the private copy is dead weight -- and dropping it keeps
    the bundle to just the fixtures still unplayed."""
    store = load_pending_props()
    for n in names:
        store.pop(n, None)
    with open(PENDING_PROPS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)

STAT_COL_BY_MARKET = {
    "Passing Yds": "passing_yards",
    "Passing TDs": "passing_tds",
    "Completions": "completions",
    "Pass Attempts": "attempts",
    "Rushing Yds": "rushing_yards",
    "Carries": "carries",
    "Receiving Yds": "receiving_yards",
    "Receptions": "receptions",
    # Without an entry here a market is projected, shown, and then NEVER
    # graded -- it silently never reaches the track record. That exact hole
    # cost the NFL props a week already.
    "FG Made": "fg_made",
    "Kicking Points": "kicking_points",
    "Rush + Rec Yds": "rush_rec_yards",
}


def load_schedule_results():
    sched = pl.read_parquet(DATA_DIR / "schedules.parquet").to_pandas()
    return sched[["season", "week", "away_team", "home_team", "away_score", "home_score"]]


def load_player_stats():
    ps = pl.read_parquet(DATA_DIR / "player_stats.parquet").to_pandas()
    ps["anytime_td"] = ((ps["rushing_tds"].fillna(0) + ps["receiving_tds"].fillna(0)) > 0)
    # Derived exactly as prop_data._load_base derives them. They are NOT
    # columns in player_stats.parquet, so without this the two markets that
    # use them would be projected and displayed but never graded, which is the
    # silent-hole failure STAT_COL_BY_MARKET exists to prevent.
    ps["rush_rec_yards"] = ps["rushing_yards"].fillna(0) + ps["receiving_yards"].fillna(0)
    ps["kicking_points"] = 3 * ps["fg_made"].fillna(0) + ps["pat_made"].fillna(0)
    ps = ps.drop_duplicates(subset=["season", "week", "player_id"])
    return ps.set_index(["season", "week", "player_id"])


def grade_prop(prop, stats_row):
    if stats_row is None:
        return {**prop, "actual_value": None, "hit": None}

    if prop["market"] == "Anytime TD":
        actual = bool(stats_row["anytime_td"])
        return {**prop, "actual_scored": actual, "hit": actual}

    col = STAT_COL_BY_MARKET.get(prop["market"])
    if col is None or col not in stats_row.index or pd.isna(stats_row[col]):
        return {**prop, "actual_value": None, "hit": None}

    actual_value = float(stats_row[col])
    line = prop["ladder"][len(prop["ladder"]) // 2]["line"] if prop.get("ladder") else prop["line"]
    return {**prop, "actual_value": actual_value, "hit": actual_value > line}


def grade_snapshot(snap, sched, player_stats, path):
    match = sched[(sched["season"] == snap["season"]) & (sched["week"] == snap["week"])
                  & (sched["away_team"] == snap["awayAbbr"]) & (sched["home_team"] == snap["homeAbbr"])]
    if match.empty or pd.isna(match.iloc[0]["home_score"]):
        return False  # not played yet

    row = match.iloc[0]
    away_score, home_score = int(row["away_score"]), int(row["home_score"])
    if home_score > away_score:
        winner = snap["homeAbbr"]
    elif away_score > home_score:
        winner = snap["awayAbbr"]
    else:
        winner = "TIE"

    model_pick = snap["homeAbbr"] if (snap["elo_home_prob"] or 0) >= 0.5 else snap["awayAbbr"]
    market_pick = None
    if snap["market_home_prob"] is not None:
        market_pick = snap["homeAbbr"] if snap["market_home_prob"] >= 0.5 else snap["awayAbbr"]

    # Props live in the git-ignored private snapshot until the game is played
    # (docs/results/ is public, and a pregame prop list there is the paid
    # product given away). Falls back to the file's own props_snapshot for
    # snapshots written before that split, so older games still grade.
    frozen_props = load_pending_props().get(path.name)
    if frozen_props is None:
        # Snapshots written before the public/private split carried their own
        # props, and still grade from them.
        frozen_props = snap.get("props_snapshot", [])
    if not frozen_props:
        print(f"  {path.name}: no frozen props found, grading game result only")

    props_graded = []
    for prop in frozen_props:
        pid = prop.get("player_id")
        key = (snap["season"], snap["week"], pid)
        stats_row = player_stats.loc[key] if pid is not None and key in player_stats.index else None
        props_graded.append(grade_prop(prop, stats_row))

    snap["graded"] = True
    snap["graded_at"] = pd.Timestamp.now().isoformat()[:19]
    snap["actual"] = {
        "away_score": away_score, "home_score": home_score, "winner": winner,
        "model_pick": model_pick, "model_correct": winner != "TIE" and model_pick == winner,
        "market_pick": market_pick,
        "market_correct": (winner != "TIE" and market_pick == winner) if market_pick else None,
    }
    snap["props_graded"] = props_graded
    return True


def main():
    if not RESULTS_DIR.exists():
        print("No docs/results/ directory yet -- nothing to grade.")
        return

    sched = load_schedule_results()
    player_stats = load_player_stats()

    files = sorted(p for p in RESULTS_DIR.glob("nfl_*.json"))
    graded_now = 0
    graded_names = []
    for path in files:
        with open(path) as f:
            snap = json.load(f)
        if snap.get("graded"):
            continue
        if grade_snapshot(snap, sched, player_stats, path):
            with open(path, "w") as f:
                json.dump(snap, f, indent=2)
            graded_now += 1
            graded_names.append(path.name)
            print(f"  graded {path.name}: {snap['actual']['winner']} won, "
                  f"model {'correct' if snap['actual']['model_correct'] else 'wrong'}")

    if graded_names:
        drop_pending_props(graded_names)
    print(f"Graded {graded_now} newly-completed games out of {len(files)} snapshot(s) on disk.")


if __name__ == "__main__":
    main()

"""Grade every player prop from finalized MLB day snapshots (docs/results/
mlb_*.json) against the real, actual boxscore stat for that game -- an
honest look at how the model's own probabilities tracked reality, not just
the team win-probability grading generate_daily_slate.py already reports.

Real per-player actuals come straight from MLB's own free boxscore endpoint
(one call per game, no key needed). A player who didn't actually appear in
the game (0 plate appearances for batters, didn't pitch for pitchers) is
excluded from the accuracy numbers and counted separately -- a "did not
play" isn't a fair test of the model's projection for what they'd do if
they did, and is a lineup-confirmation question (already handled by the
confirmed_starter badge), not a stat-projection one.

Field mapping (all direct from MLB's own boxscore, confirmed against a real
completed game -- not assumed):
  hits -> batting.hits            total_bases -> batting.totalBases
  walks -> batting.baseOnBalls    rbi -> batting.rbi
  home_runs -> batting.homeRuns > 0 (Anytime HR)
  strikeouts -> pitching.strikeOuts       hits_allowed -> pitching.hits
  walks_allowed -> pitching.baseOnBalls   runs_allowed -> pitching.runs
  outs_recorded -> pitching.outs
"""
import sys
import pathlib
import glob
import json
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.common.metrics import brier_score, log_loss, accuracy

COUNT_FIELD = {
    "Hits": ("batting", "hits"), "Total Bases": ("batting", "totalBases"),
    "Walks": ("batting", "baseOnBalls"), "RBI": ("batting", "rbi"),
    "Pitcher Strikeouts": ("pitching", "strikeOuts"), "Pitcher Hits Allowed": ("pitching", "hits"),
    "Pitcher Walks Allowed": ("pitching", "baseOnBalls"), "Pitcher Runs Allowed": ("pitching", "runs"),
    "Pitcher Outs Recorded": ("pitching", "outs"),
}
PLAYED_FIELD = {"Hits": "atBats", "Total Bases": "atBats", "Walks": "atBats", "RBI": "atBats",
                "Pitcher Strikeouts": "outs", "Pitcher Hits Allowed": "outs",
                "Pitcher Walks Allowed": "outs", "Pitcher Runs Allowed": "outs", "Pitcher Outs Recorded": "outs"}


def fetch_boxscore_stats(game_pk):
    """player_id -> {'batting': {...}, 'pitching': {...}} for every player
    who appeared in either team's boxscore."""
    resp = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    out = {}
    for side in ("home", "away"):
        for pid_key, p in data["teams"][side]["players"].items():
            pid = p["person"]["id"]
            out[pid] = p.get("stats", {})
    return out


def grade_game(g, box_by_pid):
    rows = []
    for p in g.get("props", []):
        pid = p.get("player_id")
        market = p["market"]
        if pid is None or pid not in box_by_pid:
            rows.append({"player": p["player"], "market": market, "graded": False, "reason": "not in boxscore"})
            continue
        stats = box_by_pid[pid]

        if market == "Anytime HR":
            bat = stats.get("batting", {})
            if not bat or bat.get("atBats", 0) == 0:
                rows.append({"player": p["player"], "market": market, "graded": False, "reason": "did not play"})
                continue
            actual = 1.0 if bat.get("homeRuns", 0) > 0 else 0.0
            rows.append({"player": p["player"], "market": market, "graded": True,
                         "model_prob": p["model_prob"], "actual": actual})
            continue

        section, field = COUNT_FIELD.get(market, (None, None))
        if section is None:
            continue
        block = stats.get(section, {})
        played_field = PLAYED_FIELD[market]
        if not block or block.get(played_field, 0) == 0:
            rows.append({"player": p["player"], "market": market, "graded": False, "reason": "did not play"})
            continue
        actual_val = block.get(field, 0)
        line = p["line"]
        over = 1.0 if actual_val > line else 0.0
        rows.append({"player": p["player"], "market": market, "graded": True,
                     "line": line, "actual_val": actual_val, "model_over_prob": p["model_over_prob"], "over": over})
    return rows


def main():
    snapshot_paths = sorted(glob.glob(str(ROOT / "docs" / "results" / "mlb_*.json")))
    all_rows = []
    for path in snapshot_paths:
        with open(path, encoding="utf-8") as f:
            day = json.load(f)
        if not day.get("finalized"):
            continue
        date = day["date"]
        print(f"Grading {date} ({len(day['games'])} games)...")
        for g in day["games"]:
            if not g.get("already_played") or not g.get("gamePk"):
                continue
            try:
                box = fetch_boxscore_stats(g["gamePk"])
            except Exception as e:
                print(f"  boxscore fetch failed for gamePk {g['gamePk']}: {e}")
                continue
            rows = grade_game(g, box)
            for r in rows:
                r["date"] = date
            all_rows.extend(rows)

    graded = [r for r in all_rows if r["graded"]]
    not_played = [r for r in all_rows if not r["graded"] and r["reason"] == "did not play"]
    not_found = [r for r in all_rows if not r["graded"] and r["reason"] == "not in boxscore"]

    print(f"\n=== Overall: {len(all_rows)} props seen, {len(graded)} graded "
          f"({len(not_played)} did not play, {len(not_found)} not found in boxscore) ===")

    hr_rows = [r for r in graded if r["market"] == "Anytime HR"]
    if hr_rows:
        actual = [r["actual"] for r in hr_rows]
        pred = [r["model_prob"] for r in hr_rows]
        print(f"\n--- Anytime HR (n={len(hr_rows)}) ---")
        print(f"  Brier={brier_score(actual, pred):.4f}  LogLoss={log_loss(actual, pred):.4f}  "
              f"Accuracy(>=0.5)={accuracy(actual, pred):.4f}  actual HR rate={sum(actual)/len(actual):.3f}")

    print("\n--- Count props (over/under the line) ---")
    by_market = {}
    for r in graded:
        if r["market"] == "Anytime HR":
            continue
        by_market.setdefault(r["market"], []).append(r)
    overall_over, overall_pred = [], []
    for market, rows in sorted(by_market.items()):
        actual = [r["over"] for r in rows]
        pred = [r["model_over_prob"] for r in rows]
        overall_over += actual
        overall_pred += pred
        print(f"  {market:<24} n={len(rows):>4}  Brier={brier_score(actual, pred):.4f}  "
              f"LogLoss={log_loss(actual, pred):.4f}  Accuracy={accuracy(actual, pred):.4f}  "
              f"actual-over-rate={sum(actual)/len(actual):.3f}  avg-model-prob={sum(pred)/len(pred):.3f}")
    if overall_over:
        print(f"  {'ALL COUNT PROPS':<24} n={len(overall_over):>4}  Brier={brier_score(overall_over, overall_pred):.4f}  "
              f"LogLoss={log_loss(overall_over, overall_pred):.4f}  Accuracy={accuracy(overall_over, overall_pred):.4f}")

    out_path = ROOT / "notebooks_out" / "mlb_props_live_grading.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved full row-level grading to {out_path}")


if __name__ == "__main__":
    main()

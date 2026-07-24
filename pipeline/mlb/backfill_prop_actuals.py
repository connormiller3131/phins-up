"""One-time backfill: attach real actual stat values to every prop in
already-finalized MLB day snapshots (docs/results/mlb_*.json) that predate
generate_daily_slate.py's attach_prop_actuals step. New finalizations from
here on attach this inline, so this script never needs to run against a
snapshot twice -- it's a bridge for the handful of days that were frozen
before that step existed.
"""
import sys
import pathlib
import glob
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.generate_daily_slate import attach_prop_actuals, RESULTS_DIR


def main():
    paths = sorted(glob.glob(str(RESULTS_DIR / "mlb_*.json")))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            day = json.load(f)
        if not day.get("finalized"):
            continue
        already_has_actuals = any(
            "actual" in p for g in day["games"] for p in g.get("props", [])
        )
        if already_has_actuals:
            print(f"{path}: already has actuals, skipping")
            continue
        print(f"{path}: attaching actuals for {len(day['games'])} games...")
        attach_prop_actuals(day["games"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(day, f, indent=2)
        n_graded = sum(
            1 for g in day["games"] for p in g.get("props", []) if p.get("actual") is not None
        )
        n_total = sum(len(g.get("props", [])) for g in day["games"])
        print(f"  done -- {n_graded}/{n_total} props got a real actual value")


if __name__ == "__main__":
    main()

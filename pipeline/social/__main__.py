"""CLI for the X poster.

    python -m pipeline.social --kind pregame          # dry run, prints only
    POST_TO_X=true python -m pipeline.social --kind results

Dry run is the default on purpose: the only way to post is to say so
explicitly, so an accidental run of this module can never reach the account.
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.social.post_to_x import compose_pregame, compose_results, post

NFL_PUBLISHED = ROOT / "data" / "dashboard_published_nfl.json"
RESULTS_DIR = ROOT / "docs" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pregame", "results"], required=True)
    args = ap.parse_args()

    if not NFL_PUBLISHED.exists():
        print("No published NFL data at %s -- run build_static_site.py first." % NFL_PUBLISHED)
        return 1
    nfl = json.loads(NFL_PUBLISHED.read_text(encoding="utf-8"))

    if args.kind == "pregame":
        text, key = compose_pregame(nfl)
        empty = "No unplayed game with both a model and a market number -- nothing to post."
    else:
        text, key = compose_results(nfl, RESULTS_DIR)
        empty = ("Not every game this week is graded yet -- holding, so the "
                 "posted record is the whole week rather than a partial one.")

    if not text:
        print(empty)
        return 0

    live = os.environ.get("POST_TO_X", "").lower() == "true"
    return post(text, key, live)


if __name__ == "__main__":
    sys.exit(main())

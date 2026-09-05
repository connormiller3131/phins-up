"""Build the standalone public site (docs/index.html) from the NFL and MLB
current-slate JSON + the dashboard_live.html template. Unlike the Claude
Artifact version, this keeps the full <!DOCTYPE>/<html>/<head>/<body>
document, since it's served directly by GitHub Pages rather than wrapped by
the Artifact tool. MLB data is optional -- if it hasn't been generated yet
(or the pull failed), the MLB tab just gets an empty slate rather than
failing the whole build.

Also writes an identical copy to docs/404.html -- GitHub Pages serves that
file (with a 404 status, but the content still loads and runs) for any
request path it doesn't recognize, which is exactly what a client-side
route like /mlb/monday looks like to it. Keeping this as a byte-identical
copy generated every run means it can never drift from index.html; the
Cloudflare side of this (site-worker.js) solves the same problem its own
way, since GitHub Pages' 404.html convention doesn't apply there."""
import pathlib
import sys
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.common.gated_payload import build_gated_payload
NFL_DATA_PATH = ROOT / "data" / "nfl" / "dashboard_current_week.json"
MLB_DATA_PATH = ROOT / "data" / "mlb" / "dashboard_current_slate.json"
NHL_DATA_PATH = ROOT / "data" / "nhl" / "dashboard_current_slate.json"
TEMPLATE_PATH = ROOT / "pipeline" / "nfl" / "dashboard_live.html"
OUT_PATH = ROOT / "docs" / "index.html"
NOT_FOUND_PATH = ROOT / "docs" / "404.html"
# Deliberately under data/ (git-ignored) and NOT docs/: anything written to
# docs/ is committed and then served publicly twice over, once by Cloudflare
# and once by GitHub Pages straight off the repo. refresh.yml uploads this
# file to Cloudflare KV instead, which is the only path a browser can reach
# it by.
GATED_PATH = ROOT / "data" / "gated_payload.json"

_EMPTY_DAY_SLATE = {"week_start": None, "week_end": None, "today": None, "generated_at": None, "days": {}}


def main():
    with open(NFL_DATA_PATH, encoding="utf-8") as f:
        nfl_data = json.load(f)

    if MLB_DATA_PATH.exists():
        with open(MLB_DATA_PATH, encoding="utf-8") as f:
            mlb_data = json.load(f)
    else:
        mlb_data = dict(_EMPTY_DAY_SLATE)

    if NHL_DATA_PATH.exists():
        with open(NHL_DATA_PATH, encoding="utf-8") as f:
            nhl_data = json.load(f)
    else:
        nhl_data = dict(_EMPTY_DAY_SLATE)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        tmpl = f.read()

    # Lift the paid half out BEFORE anything is serialised into the page --
    # nfl_data/mlb_data are mutated into their free form here, so there is no
    # way to accidentally inline the gated fields further down.
    gated = build_gated_payload(nfl_data, mlb_data)
    GATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GATED_PATH, "w", encoding="utf-8") as f:
        json.dump(gated, f)

    out = (tmpl.replace("__DATA_JSON__", json.dumps(nfl_data))
               .replace("__MLB_DATA_JSON__", json.dumps(mlb_data))
               .replace("__NHL_DATA_JSON__", json.dumps(nhl_data)))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    with open(NOT_FOUND_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Free page {OUT_PATH.stat().st_size/1e6:.2f} MB inlined; "
          f"gated payload {GATED_PATH.stat().st_size/1e6:.2f} MB held back "
          f"({len(gated['nfl'])} NFL games, {len(gated['mlb'])} MLB games)")
    print(f"Built {OUT_PATH} -- NFL season {nfl_data['season']} current week {nfl_data['current_week']}, "
          f"MLB week {mlb_data.get('week_start')} to {mlb_data.get('week_end')} (today={mlb_data.get('today')}), "
          f"NHL week {nhl_data.get('week_start')} to {nhl_data.get('week_end')} (today={nhl_data.get('today')})")


if __name__ == "__main__":
    main()

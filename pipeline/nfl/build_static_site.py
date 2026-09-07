"""Build the standalone public site (docs/index.html) from the NFL and MLB
current-slate JSON + the dashboard_live.html template. Keeps the full
<!DOCTYPE>/<html>/<head>/<body> document, since it is served directly rather
than wrapped by anything. MLB data is optional -- if it hasn't been generated
yet (or the pull failed), the MLB tab just gets an empty slate rather than
failing the whole build.

This used to also write a byte-identical docs/404.html for GitHub Pages,
which serves that file (with a 404 status, but the content still loads) for
any path it doesn't recognise -- which is what a client-side route like
/mlb/monday looks like to it. GitHub Pages was disabled on 2026-09-06, so
that copy is gone: it was 0.9 MB committed twice a day for a convention
nothing uses any more. Cloudflare solves the same routing problem its own
way, via wrangler.toml's not_found_handling and site-worker.js's own
404 -> index.html fallback."""
import datetime
import pathlib
import sys
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.common.gated_payload import build_gated_payload
from pipeline.nfl import build_game_pages
NFL_DATA_PATH = ROOT / "data" / "nfl" / "dashboard_current_week.json"
MLB_DATA_PATH = ROOT / "data" / "mlb" / "dashboard_current_slate.json"
NHL_DATA_PATH = ROOT / "data" / "nhl" / "dashboard_current_slate.json"
TEMPLATE_PATH = ROOT / "pipeline" / "nfl" / "dashboard_live.html"
OUT_PATH = ROOT / "docs" / "index.html"
# Deliberately under data/ (git-ignored) and NOT docs/: anything written to
# docs/ is committed and then served publicly twice over, once by Cloudflare
# and once by GitHub Pages straight off the repo. refresh.yml uploads this
# file to Cloudflare KV instead, which is the only path a browser can reach
# it by.
GATED_PATH = ROOT / "data" / "gated_payload.json"
# The NFL data as actually PUBLISHED, after future weeks are dropped. The Node
# scripts that build and verify the account payload read this rather than the
# raw pipeline output, so "what got published" has one definition instead of
# the same trim being reimplemented in two languages.
PUBLISHED_NFL_PATH = ROOT / "data" / "dashboard_published_nfl.json"

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

    # Future weeks are not published at all. Hiding the week dropdown would
    # not do: every future week's games and props would still be sitting in
    # the page source and the gated payload for anyone who looked. On a real
    # board that is 256 games and 12,448 props -- 94% of everything published
    # -- so dropping them is both the privacy fix and by far the largest size
    # win available. Past weeks stay: they are completed games and they are
    # what the track record is built from.
    current_week = int(nfl_data["current_week"])
    dropped = [w for w in nfl_data["weeks"] if int(w) > current_week]
    nfl_data["weeks"] = {w: v for w, v in nfl_data["weeks"].items() if int(w) <= current_week}
    with open(PUBLISHED_NFL_PATH, "w", encoding="utf-8") as f:
        json.dump(nfl_data, f)
    print(f"Publishing weeks <= {current_week}; withheld {len(dropped)} future week(s)")

    # Lift the paid half out BEFORE anything is serialised into the page --
    # nfl_data/mlb_data are mutated into their free form here, so there is no
    # way to accidentally inline the gated fields further down.
    gated = build_gated_payload(nfl_data, mlb_data)
    GATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GATED_PATH, "w", encoding="utf-8") as f:
        json.dump(gated, f)

    # AFTER build_gated_payload, which has already popped `props` off every
    # game -- so the dicts handed to the page builder physically cannot carry
    # paid data into a publicly served file. Reordering those two would turn a
    # safe-by-construction guarantee into a review problem.
    #
    # And BEFORE the template is serialised, because the builder stamps a
    # `page_url` onto each game as it writes that game's page. The page then
    # reads that field instead of recomputing the slug in JavaScript. The two
    # rules are not the same -- NFL takes the last word of a team name, MLB
    # has to strip a city prefix or the Red Sox and the White Sox both come
    # out as "sox" -- so having one source of truth is worth the ordering
    # constraint.
    build_game_pages.build(nfl_data, mlb_data, OUT_PATH.parent,
                           OUT_PATH.parent / "results",
                           datetime.date.today().isoformat())

    out = (tmpl.replace("__DATA_JSON__", json.dumps(nfl_data))
               .replace("__MLB_DATA_JSON__", json.dumps(mlb_data))
               .replace("__NHL_DATA_JSON__", json.dumps(nhl_data)))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Free page {OUT_PATH.stat().st_size/1e6:.2f} MB inlined; "
          f"gated payload {GATED_PATH.stat().st_size/1e6:.2f} MB held back "
          f"({len(gated['nfl'])} NFL games, {len(gated['mlb'])} MLB games)")
    print(f"Built {OUT_PATH} -- NFL season {nfl_data['season']} current week {nfl_data['current_week']}, "
          f"MLB week {mlb_data.get('week_start')} to {mlb_data.get('week_end')} (today={mlb_data.get('today')}), "
          f"NHL week {nhl_data.get('week_start')} to {nhl_data.get('week_end')} (today={nhl_data.get('today')})")


if __name__ == "__main__":
    main()

"""Build the standalone public site (docs/index.html) from the NFL and MLB
current-slate JSON + the dashboard_live.html template. Unlike the Claude
Artifact version, this keeps the full <!DOCTYPE>/<html>/<head>/<body>
document, since it's served directly by GitHub Pages rather than wrapped by
the Artifact tool. MLB data is optional -- if it hasn't been generated yet
(or the pull failed), the MLB tab just gets an empty slate rather than
failing the whole build."""
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
NFL_DATA_PATH = ROOT / "data" / "nfl" / "dashboard_current_week.json"
MLB_DATA_PATH = ROOT / "data" / "mlb" / "dashboard_current_slate.json"
NHL_DATA_PATH = ROOT / "data" / "nhl" / "dashboard_current_slate.json"
TEMPLATE_PATH = ROOT / "pipeline" / "nfl" / "dashboard_live.html"
OUT_PATH = ROOT / "docs" / "index.html"

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

    out = (tmpl.replace("__DATA_JSON__", json.dumps(nfl_data))
               .replace("__MLB_DATA_JSON__", json.dumps(mlb_data))
               .replace("__NHL_DATA_JSON__", json.dumps(nhl_data)))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Built {OUT_PATH} -- NFL season {nfl_data['season']} current week {nfl_data['current_week']}, "
          f"MLB week {mlb_data.get('week_start')} to {mlb_data.get('week_end')} (today={mlb_data.get('today')}), "
          f"NHL week {nhl_data.get('week_start')} to {nhl_data.get('week_end')} (today={nhl_data.get('today')})")


if __name__ == "__main__":
    main()

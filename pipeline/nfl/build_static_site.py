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
TEMPLATE_PATH = ROOT / "pipeline" / "nfl" / "dashboard_live.html"
OUT_PATH = ROOT / "docs" / "index.html"


def main():
    with open(NFL_DATA_PATH, encoding="utf-8") as f:
        nfl_data = json.load(f)

    if MLB_DATA_PATH.exists():
        with open(MLB_DATA_PATH, encoding="utf-8") as f:
            mlb_data = json.load(f)
    else:
        mlb_data = {"date": None, "elo_params": {"k": None, "home_adv": None, "scale": None},
                    "generated_at": None, "games": []}

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        tmpl = f.read()

    out = tmpl.replace("__DATA_JSON__", json.dumps(nfl_data)).replace("__MLB_DATA_JSON__", json.dumps(mlb_data))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Built {OUT_PATH} -- NFL season {nfl_data['season']} week {nfl_data['week']}, MLB date {mlb_data['date']}")


if __name__ == "__main__":
    main()

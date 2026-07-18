"""Build the standalone public site (docs/index.html) from the current-week
JSON + the dashboard_live.html template. Unlike the Claude Artifact version,
this keeps the full <!DOCTYPE>/<html>/<head>/<body> document, since it's
served directly by GitHub Pages rather than wrapped by the Artifact tool."""
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "nfl" / "dashboard_current_week.json"
TEMPLATE_PATH = ROOT / "pipeline" / "nfl" / "dashboard_live.html"
OUT_PATH = ROOT / "docs" / "index.html"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        tmpl = f.read()

    out = tmpl.replace("__DATA_JSON__", json.dumps(data))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Built {OUT_PATH} for season {data['season']} week {data['week']}")


if __name__ == "__main__":
    main()

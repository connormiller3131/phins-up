"""Pull real per-game starting-goalie boxscore stats for every regular
season game already in team_games.parquet -- the schedule endpoint that
pipeline already pulls from (api-web.nhle.com/v1/schedule) doesn't return
player-level detail at all, so this hits a different, real endpoint
(v1/gamecenter/{id}/boxscore, confirmed directly: has a per-goalie
`starter: true/false` flag, save%, goals against, shots against) -- a
genuinely new data source for this pipeline, not a re-parse of something
already pulled.

Two-step process, both resumable/incremental (safe to interrupt and rerun):
1. Real game IDs aren't in team_games.parquet at all (that table only has
   date/teams/scores from the schedule endpoint) -- get them from
   club-schedule-season/{team}/{season}, one call per (team, season),
   which returns that team's whole season including the real game id.
   Only ~256 calls total (32 teams x 8 seasons), each covering ~41-101
   games at once -- cheap enough to redo in full every run, so this half
   is never cached to disk.
2. One boxscore call per UNIQUE real game id (~9,800 games across 8
   seasons) to get both teams' starting goalie stats for that game. This
   half is genuinely expensive (~25hrs observed for the initial full
   backfill, real rate-limiting) -- data/nhl/goalie_game_logs.parquet is
   therefore committed to git as a one-time seed (a deliberate, narrow
   .gitignore exception -- see the comment there) rather than left to a
   GitHub Actions cache that could ever go cold. Every run only attempts
   games not already in that file AND already played (game_date <= now),
   so ongoing CI runs pull a handful of newly-completed games at most,
   fast, then commit the small delta back (refresh.yml, same step that
   commits docs/index.html).
"""
import sys
import pathlib
import time
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nhl.team_map import CURRENT_TEAMS, normalize_team

DATA_DIR = ROOT / "data" / "nhl"
GAME_ID_CACHE = DATA_DIR / "game_ids.parquet"
GOALIE_LOG_PATH = DATA_DIR / "goalie_game_logs.parquet"
SEASONS = list(range(2018, 2026))  # start years, matches team_games.parquet's real coverage
CHECKPOINT_EVERY = 200
REQUEST_DELAY = 0.5  # real rate limiting confirmed on a first attempt with no delay at all (429s within the first ~30 requests)

session = requests.Session()


def get_with_retry(url, max_retries=6):
    """A real 429 (confirmed directly -- this API rate-limits hard with no
    delay between requests) needs actual backoff, not just a fixed sleep
    per request -- honors Retry-After when the server sends one."""
    for attempt in range(max_retries):
        resp = session.get(url, timeout=15)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = float(resp.headers.get("Retry-After", 2 ** attempt))
        time.sleep(wait)
    resp.raise_for_status()
    return resp


def fetch_game_ids():
    if GAME_ID_CACHE.exists():
        print(f"Reusing cached game IDs: {GAME_ID_CACHE}")
        return pd.read_parquet(GAME_ID_CACHE)

    rows = []
    for team in sorted(CURRENT_TEAMS):
        for season in SEASONS:
            season_str = f"{season}{season+1}"
            try:
                resp = get_with_retry(f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season_str}")
                data = resp.json()
            except Exception as e:
                print(f"  {team} {season_str}: fetch failed ({e}), skipping")
                continue
            time.sleep(REQUEST_DELAY)
            for g in data.get("games", []):
                if g.get("gameType") != 2:
                    continue
                rows.append({
                    "game_id": g["id"], "game_date": g["gameDate"],
                    "away_team": normalize_team(g["awayTeam"]["abbrev"]), "home_team": normalize_team(g["homeTeam"]["abbrev"]),
                })
        print(f"  {team}: done")

    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GAME_ID_CACHE)
    print(f"Wrote {len(df)} unique real game IDs to {GAME_ID_CACHE}")
    return df


def fetch_goalie_rows(game_id):
    resp = get_with_retry(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore")
    box = resp.json()
    pbg = box.get("playerByGameStats", {})
    out = []
    for side, team_key in (("awayTeam", "away"), ("homeTeam", "home")):
        team_abbr = normalize_team(box.get(team_key + "Team", {}).get("abbrev", ""))
        for g in pbg.get(side, {}).get("goalies", []):
            if g.get("toi", "0:00") == "0:00":
                continue  # dressed but didn't play (backup who never entered)
            out.append({
                "game_id": game_id, "team": team_abbr, "player_id": g["playerId"],
                "starter": bool(g.get("starter", False)),
                "shots_against": g.get("shotsAgainst"), "saves": g.get("saves"),
                "goals_against": g.get("goalsAgainst"), "save_pctg": g.get("savePctg"),
            })
    return out


def main():
    game_ids_df = fetch_game_ids()

    already_done = set()
    existing = []
    if GOALIE_LOG_PATH.exists():
        prev = pd.read_parquet(GOALIE_LOG_PATH)
        already_done = set(prev["game_id"].unique())
        existing = [prev]
        print(f"Resuming: {len(already_done)} games already pulled")

    not_done = ~game_ids_df["game_id"].isin(already_done)
    already_played = pd.to_datetime(game_ids_df["game_date"]) <= pd.Timestamp.now()
    todo = game_ids_df[not_done & already_played]
    print(f"{len(todo)} already-played games left to pull "
          f"({(not_done & ~already_played).sum()} more scheduled but not played yet, skipped for now)")

    buffer = []
    done_count = 0
    for row in todo.itertuples(index=False):
        try:
            rows = fetch_goalie_rows(row.game_id)
            for r in rows:
                r["game_date"] = row.game_date
            buffer.extend(rows)
        except Exception as e:
            print(f"  game {row.game_id} failed: {e}")
        time.sleep(REQUEST_DELAY)
        done_count += 1
        if done_count % CHECKPOINT_EVERY == 0:
            existing = _checkpoint(existing, buffer)
            buffer = []
            print(f"  checkpointed at {done_count}/{len(todo)}")

    if buffer:
        existing = _checkpoint(existing, buffer)

    final = pd.read_parquet(GOALIE_LOG_PATH) if GOALIE_LOG_PATH.exists() else pd.concat(existing, ignore_index=True)
    print(f"\nDone. {len(final)} goalie-game rows across {final['game_id'].nunique()} games.")


def _checkpoint(existing_frames, buffer_rows):
    """Writes to disk and returns the new in-memory frame list to carry
    forward -- never re-reads the file back (that crashed once already
    when an entire batch failed and the file was never created)."""
    if not buffer_rows:
        return existing_frames
    new = pd.DataFrame(buffer_rows)
    combined = pd.concat(existing_frames + [new], ignore_index=True) if existing_frames else new
    combined.to_parquet(GOALIE_LOG_PATH)
    return [combined]


if __name__ == "__main__":
    main()

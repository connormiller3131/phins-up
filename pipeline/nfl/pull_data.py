"""Pull NFL schedules, play-by-play, and player stats via nflreadpy and cache locally as parquet."""
import io
import pathlib
import datetime
import urllib.request

import nflreadpy as nfl
import polars as pl

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nfl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Upper bound tracks the current calendar year so in-progress/future seasons
# (e.g. 2026) are always included once their games start completing --
# nflreadpy safely returns whatever rows exist for a season, played or not.
SEASONS = list(range(2019, datetime.date.today().year + 1))


# nflreadpy refuses a season until the Thursday following Labor Day, and its
# range check is hard ("Season must be between 2009 and 2025"). The NFL does
# not agree: in 2026 the opener is WEDNESDAY Sept 9, a day before nflreadpy
# considers the season to exist. In most years the opener IS that Thursday and
# nobody notices, which is exactly why this went unseen.
#
# The consequence was not cosmetic. injuries.parquet had zero 2026 rows, so
# load_injury_status() found nothing, the "ruled Out" filter became a silent
# no-op, and TreVeyon Henderson -- listed Out (Ankle) by nflverse for NE week 1
# -- was still being projected for the game he was already ruled out of.
#
# nflverse publishes the report regardless, so this reads the release asset
# directly for any season the library will not serve. Genuinely absent seasons
# 404 and are skipped, which is the correct outcome for a season that has not
# started: it is only the injury report that exists before any game is played.
INJURY_RELEASE = ("https://github.com/nflverse/nflverse-data/releases/download/"
                  "injuries/injuries_%d.parquet")


def load_injuries_including_current(seasons):
    """load_injuries, plus any season nflreadpy declines but nflverse has."""
    supported = [s for s in seasons if s <= nfl.get_current_season()]
    df = nfl.load_injuries(seasons=supported) if supported else None

    for season in [s for s in seasons if s > nfl.get_current_season()]:
        try:
            with urllib.request.urlopen(INJURY_RELEASE % season, timeout=60) as r:
                extra = pl.read_parquet(io.BytesIO(r.read()))
        except Exception as e:
            print(f"  injuries {season}: not published yet ({type(e).__name__}); skipping")
            continue
        if extra.is_empty():
            continue
        n_out = extra.filter(pl.col("report_status") == "Out").height
        print(f"  injuries {season}: {extra.height} rows pulled directly from "
              f"nflverse ({n_out} ruled Out) -- nflreadpy will not serve this "
              f"season until the Thursday after Labor Day")
        if df is None:
            df = extra
        else:
            # Column sets can differ slightly between seasons; align on the
            # shared ones rather than throwing away the new season entirely.
            shared = [c for c in df.columns if c in extra.columns]
            # vertical_relaxed, not vertical: nflverse types a fresh season's
            # columns tighter than the historical frames (season is Int32 here
            # against Float64 there), and a strict concat rejects that.
            df = pl.concat([df.select(shared), extra.select(shared)],
                           how="vertical_relaxed")
    return df


def save(df, name):
    path = DATA_DIR / f"{name}.parquet"
    df.write_parquet(path)
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols -> {path}")


def main():
    print(f"Pulling seasons {SEASONS[0]}-{SEASONS[-1]}...")

    schedules = nfl.load_schedules(seasons=SEASONS)  # schedules includes future/in-progress seasons
    save(schedules, "schedules")

    # Team conf/division for the standings/season view -- not tied to a
    # season, just the current alignment, so no seasons param.
    teams = nfl.load_teams()
    save(teams, "teams")

    # Play-by-play, player/team stats, and rosters only exist for seasons with
    # played games -- nflreadpy rejects anything past its own "current season"
    # (which tracks completed data), so cap those pulls there.
    stats_seasons = [s for s in SEASONS if s <= nfl.get_current_season()]
    print(f"Stats/pbp/roster seasons capped at {stats_seasons[-1]} (nflreadpy's current completed season)")

    pbp = nfl.load_pbp(seasons=stats_seasons)
    save(pbp, "pbp")

    player_stats = nfl.load_player_stats(seasons=stats_seasons)
    save(player_stats, "player_stats")

    team_stats = nfl.load_team_stats(seasons=stats_seasons)
    save(team_stats, "team_stats")

    rosters = nfl.load_rosters_weekly(seasons=stats_seasons)
    save(rosters, "rosters_weekly")

    # Weekly injury report: per player per week, with report_status
    # (Out/Doubtful/Questionable/None) and gsis_id, which joins directly to
    # player_stats' player_id and the depth chart's gsis_id (verified: 1377
    # of 1453 2025 injury ids match player_stats -- the misses are defensive
    # players who never appear in offensive player stats, as expected).
    #
    # Two distinct uses, deliberately kept separate: filtering props for
    # players ruled Out is a pure data fix with no modelling risk, while
    # feeding injuries into win probability is a model change that has to
    # earn its place in a backtest first.
    # SEASONS, not stats_seasons: the injury report is the one feed that
    # exists before a season's first game is played, and the current
    # season is precisely the one we need it for.
    injuries = load_injuries_including_current(SEASONS)
    save(injuries, "injuries")

    print("Done.")


if __name__ == "__main__":
    main()

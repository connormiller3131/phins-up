"""Pull NFL schedules, play-by-play, and player stats via nflreadpy and cache locally as parquet."""
import pathlib
import datetime
import nflreadpy as nfl

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nfl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Upper bound tracks the current calendar year so in-progress/future seasons
# (e.g. 2026) are always included once their games start completing --
# nflreadpy safely returns whatever rows exist for a season, played or not.
SEASONS = list(range(2019, datetime.date.today().year + 1))


def save(df, name):
    path = DATA_DIR / f"{name}.parquet"
    df.write_parquet(path)
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols -> {path}")


def main():
    print(f"Pulling seasons {SEASONS[0]}-{SEASONS[-1]}...")

    schedules = nfl.load_schedules(seasons=SEASONS)  # schedules includes future/in-progress seasons
    save(schedules, "schedules")

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

    print("Done.")


if __name__ == "__main__":
    main()

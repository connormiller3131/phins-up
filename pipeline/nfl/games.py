"""Build the core chronological game table used by all NFL win-probability models."""
import pathlib
import numpy as np
import polars as pl

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nfl"


def moneyline_to_prob(ml):
    """Convert American odds to raw (vig-included) implied probability."""
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    return prob


def load_games():
    sched = pl.read_parquet(DATA_DIR / "schedules.parquet")
    games = (
        sched.filter(pl.col("home_score").is_not_null())
        .select([
            "game_id", "season", "week", "game_type", "gameday", "away_team", "home_team",
            "away_score", "home_score", "location",
            "spread_line", "total_line", "away_moneyline", "home_moneyline",
            "away_rest", "home_rest", "roof", "temp", "wind",
        ])
        .with_columns([
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
            pl.when(pl.col("home_score") > pl.col("away_score")).then(1.0)
              .when(pl.col("home_score") < pl.col("away_score")).then(0.0)
              .otherwise(0.5).alias("home_win"),
            pl.col("gameday").str.to_date().alias("game_date"),
        ])
        .sort(["game_date", "week"])
    )

    df = games.to_pandas()
    df["away_ml_prob_raw"] = moneyline_to_prob(df["away_moneyline"].values)
    df["home_ml_prob_raw"] = moneyline_to_prob(df["home_moneyline"].values)
    overround = df["away_ml_prob_raw"] + df["home_ml_prob_raw"]
    df["market_home_prob"] = df["home_ml_prob_raw"] / overround  # no-vig normalization
    return df


if __name__ == "__main__":
    df = load_games()
    print(df.shape)
    print(df[["season", "week", "home_team", "away_team", "home_win", "market_home_prob"]].head(10))

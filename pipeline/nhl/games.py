"""Load the cached NHL game table and attach per-team rest days and trailing
goal-scoring form, walk-forward safe (shift(1) before any rolling/diff, so a
team's own current game never leaks into its own features) -- same pattern
as MLB's games.py."""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "nhl"
FORM_WINDOW = 10
FORM_MIN_GAMES = 3


def _team_rest_and_form(games):
    """Long table, one row per team per game -- a team's away games count
    toward its rest/form just as much as its home games do."""
    home_side = pd.DataFrame({
        "season": games["season"], "game_date": games["game_date"],
        "team": games["home_team"], "goals_scored": games["home_score"], "goals_allowed": games["away_score"],
    })
    away_side = pd.DataFrame({
        "season": games["season"], "game_date": games["game_date"],
        "team": games["away_team"], "goals_scored": games["away_score"], "goals_allowed": games["home_score"],
    })
    long = pd.concat([home_side, away_side], ignore_index=True).sort_values(["team", "game_date"])

    long["prev_game_date"] = long.groupby("team")["game_date"].shift(1)
    long["rest_days"] = (long["game_date"] - long["prev_game_date"]).dt.days

    long["trailing_goals_scored"] = long.groupby("team")["goals_scored"].transform(
        lambda s: s.shift(1).rolling(window=FORM_WINDOW, min_periods=FORM_MIN_GAMES).mean()
    )
    long["trailing_goals_allowed"] = long.groupby("team")["goals_allowed"].transform(
        lambda s: s.shift(1).rolling(window=FORM_WINDOW, min_periods=FORM_MIN_GAMES).mean()
    )
    return long[["team", "game_date", "rest_days", "trailing_goals_scored", "trailing_goals_allowed"]]


def load_games():
    games = pd.read_parquet(DATA_DIR / "team_games.parquet")
    games["margin"] = games["home_score"] - games["away_score"]
    games["home_win"] = (games["margin"] > 0).astype(float)
    games = games.sort_values("game_date").reset_index(drop=True)

    # A team never plays twice on the same calendar date in the NHL (unlike
    # MLB doubleheaders), so a plain (team, game_date) join is safe here --
    # no occurrence-disambiguation needed.
    form = _team_rest_and_form(games)

    games = games.merge(
        form.rename(columns={"team": "home_team", "rest_days": "home_rest",
                              "trailing_goals_scored": "home_trailing_goals_scored",
                              "trailing_goals_allowed": "home_trailing_goals_allowed"}),
        on=["home_team", "game_date"], how="left",
    )
    games = games.merge(
        form.rename(columns={"team": "away_team", "rest_days": "away_rest",
                              "trailing_goals_scored": "away_trailing_goals_scored",
                              "trailing_goals_allowed": "away_trailing_goals_allowed"}),
        on=["away_team", "game_date"], how="left",
    )
    return games


if __name__ == "__main__":
    df = load_games()
    print(df.shape)
    print(df["season"].value_counts().sort_index())
    print(df.tail(10))

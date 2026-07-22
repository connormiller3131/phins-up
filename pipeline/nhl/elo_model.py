"""Margin-of-victory Elo rating model for NHL. Same architecture as the
MLB/NFL models -- a log-margin multiplier rather than a sport-specific
published formula, with the overall sensitivity left entirely to the fitted
K. All four knobs -- K, home-advantage, scale, and between-season
regression -- are grid-searched on real data, not assumed.

NHL-specific note: goal margins are small and a meaningful fraction of games
are decided by a single goal in overtime or a shootout, which isn't really
comparable in quality to a single-goal regulation win -- the log(margin+1)
multiplier already compresses this (log(2) vs log(1) isn't a huge gap), and
whether a sport-specific OT/SO discount improves on that is left as an open
question for backtest_win_prob.py to test empirically, not assumed here."""
import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0


def run_elo(df: pd.DataFrame, k: float, home_adv: float, scale: float, season_regression: float = 0.65):
    ratings = {}
    last_season = {}
    preds = np.zeros(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        home, away, season = row.home_team, row.away_team, row.season
        for team in (home, away):
            if team not in ratings:
                ratings[team] = INITIAL_RATING
                last_season[team] = season
            elif last_season[team] != season:
                ratings[team] = season_regression * ratings[team] + (1 - season_regression) * INITIAL_RATING
                last_season[team] = season

        r_home, r_away = ratings[home], ratings[away]
        diff = (r_home + home_adv) - r_away
        p_home = 1.0 / (1.0 + 10 ** (-diff / scale))
        preds[i] = p_home

        if pd.isna(row.margin):
            continue

        mov_mult = np.log(abs(row.margin) + 1)
        actual = row.home_win
        delta = k * mov_mult * (actual - p_home)
        ratings[home] += delta
        ratings[away] -= delta

    return preds


def fit_elo_hyperparams(train_df: pd.DataFrame):
    """Grid search over K, home advantage, scale, and between-season rating
    regression -- same ranges as MLB's, widened there after an initial grid
    landed on boundary values; starting from that already-widened range here
    rather than repeating the same mistake."""
    from pipeline.common.metrics import log_loss

    best = None
    for k in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 14, 18, 24):
        for home_adv in (0, 10, 20, 30, 40, 50, 60):
            for scale in (150, 200, 250, 300, 350, 400, 450, 500, 600):
                for season_regression in (0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9):
                    preds = run_elo(train_df, k=k, home_adv=home_adv, scale=scale,
                                    season_regression=season_regression)
                    ll = log_loss(train_df["home_win"], preds)
                    if best is None or ll < best[0]:
                        best = (ll, k, home_adv, scale, season_regression)

    ll, k, home_adv, scale, season_regression = best
    return {"k": k, "home_adv": home_adv, "scale": scale,
            "season_regression": season_regression, "train_log_loss": ll}

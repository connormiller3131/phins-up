"""Margin-of-victory Elo rating model for MLB. Same architecture as the NFL
model (pipeline/nfl/elo_model.py), but with a simpler log-margin multiplier
instead of NFL's borrowed point-scale constants -- baseball run margins are
small and don't need a sport-specific published formula, so the overall
sensitivity is left entirely to the fitted K. K / home-advantage / scale are
grid-searched on real data, not assumed."""
import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0
SEASON_REGRESSION = 0.65  # MLB rosters/form churn more between seasons than NFL


def run_elo(df: pd.DataFrame, k: float, home_adv: float, scale: float):
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
                ratings[team] = SEASON_REGRESSION * ratings[team] + (1 - SEASON_REGRESSION) * INITIAL_RATING
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
    from pipeline.common.metrics import log_loss

    best = None
    for k in (4, 6, 8, 10, 14, 18, 24):
        for home_adv in (0, 10, 20, 30, 40):
            for scale in (200, 250, 300, 350):
                preds = run_elo(train_df, k=k, home_adv=home_adv, scale=scale)
                ll = log_loss(train_df["home_win"], preds)
                if best is None or ll < best[0]:
                    best = (ll, k, home_adv, scale)

    ll, k, home_adv, scale = best
    return {"k": k, "home_adv": home_adv, "scale": scale, "train_log_loss": ll}

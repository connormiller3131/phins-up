"""Margin-of-victory Elo rating model for NFL, with K / home-advantage / scale
fit by grid search on training seasons (not assumed constants)."""
import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0
SEASON_REGRESSION = 0.75  # fraction of rating carried over between seasons; rest reverts to 1500


def run_elo(df: pd.DataFrame, k: float, home_adv: float, scale: float, rest_adv: float = 0.0):
    """Sequentially simulate Elo through df (must be chronologically sorted).
    Returns array of pre-game home win probabilities, aligned to df rows."""
    ratings = {}
    last_season = {}
    preds = np.zeros(len(df))
    has_rest = "home_rest" in df.columns and "away_rest" in df.columns

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
        neutral = row.location == "Neutral" if hasattr(row, "location") else False
        adj_home_adv = 0.0 if neutral else home_adv
        rest_bonus = 0.0
        if has_rest and pd.notna(row.home_rest) and pd.notna(row.away_rest):
            rest_bonus = rest_adv * (row.home_rest - row.away_rest)
        diff = (r_home + adj_home_adv + rest_bonus) - r_away
        p_home = 1.0 / (1.0 + 10 ** (-diff / scale))
        preds[i] = p_home

        if pd.isna(row.margin):
            continue  # future/unplayed game: record the prediction, nothing to update ratings with yet

        margin = abs(row.margin)
        elo_diff_winner = diff if row.margin >= 0 else -diff  # pre-game rating edge of the actual winner
        mov_mult = ((margin + 3) ** 0.8) / max(7.5 + 0.006 * elo_diff_winner, 1.0)

        actual = row.home_win  # 1, 0, or 0.5 for tie
        delta = k * mov_mult * (actual - p_home)
        ratings[home] += delta
        ratings[away] -= delta

    return preds


def fit_elo_hyperparams(train_df: pd.DataFrame):
    """Coarse grid search over K, home_adv, scale, rest_adv minimizing log loss on train_df only."""
    from pipeline.common.metrics import log_loss

    best = None
    for k in (8, 12, 16, 20, 26, 32):
        for home_adv in (0, 25, 40, 55, 70, 90):
            for scale in (300, 350, 400, 450):
                for rest_adv in (0, 3, 6, 9, 12, 15):
                    preds = run_elo(train_df, k=k, home_adv=home_adv, scale=scale, rest_adv=rest_adv)
                    mask = train_df["home_win"] != 0.5  # exclude ties from loss calc
                    ll = log_loss(train_df.loc[mask, "home_win"], preds[mask.values])
                    if best is None or ll < best[0]:
                        best = (ll, k, home_adv, scale, rest_adv)

    ll, k, home_adv, scale, rest_adv = best
    return {"k": k, "home_adv": home_adv, "scale": scale, "rest_adv": rest_adv, "train_log_loss": ll}

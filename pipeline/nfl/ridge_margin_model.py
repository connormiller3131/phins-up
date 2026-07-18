"""Ridge-regression team power ratings: regress score margin on home/away team
dummies (+home-field intercept), refit walk-forward so every prediction only uses
strictly earlier games. Alpha is chosen by RidgeCV (leave-one-out) each refit,
not hand-picked. Margin -> win probability via a Normal CDF scaled by in-sample
residual std."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import RidgeCV


def _design_matrix(df: pd.DataFrame, teams: list[str]):
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(df)
    X = np.zeros((n, len(teams) + 1))  # + home-field intercept column
    X[:, 0] = 1.0
    for row_i, (home, away) in enumerate(zip(df["home_team"], df["away_team"])):
        X[row_i, 1 + team_idx[home]] = 1.0
        X[row_i, 1 + team_idx[away]] = -1.0
    return X


def walk_forward_ridge(df: pd.DataFrame, test_seasons: list[int]):
    """For each (season, week) in test_seasons, fit on all strictly earlier games,
    predict that week's margins, return win-prob predictions aligned to df rows
    (NaN for rows outside test_seasons)."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    preds = np.full(len(df), np.nan)

    test_mask = df["season"].isin(test_seasons)
    test_keys = df.loc[test_mask, ["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    for season, week in test_keys.itertuples(index=False):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        target_idx = df.index[(df["season"] == season) & (df["week"] == week)]
        if len(train) < 50:  # not enough history yet to fit sensibly
            continue

        X_train = _design_matrix(train, teams)
        y_train = train["margin"].values
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(X_train, y_train)

        resid_std = float(np.std(y_train - model.predict(X_train)))
        resid_std = max(resid_std, 1e-6)

        X_test = _design_matrix(df.loc[target_idx], teams)
        pred_margin = model.predict(X_test)
        p_home = norm.cdf(pred_margin / resid_std)
        preds[df.index.get_indexer(target_idx)] = p_home

    return preds

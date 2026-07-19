"""Walk-forward models for player props:
- yardage props (continuous): RidgeCV regression on [own_trailing_avg, opp_allowed_trailing_avg]
- anytime-TD (binary): LogisticRegressionCV on [own_trailing_avg, opp_allowed_trailing_avg]
Both refit once per test week using only strictly earlier rows -> no leakage.
Alpha/C are chosen by internal cross-validation, not hand-picked."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import RidgeCV, LogisticRegressionCV


FEATURES = ["own_trailing_avg", "opp_allowed_trailing_avg", "is_dome", "temp", "wind", "own_rest", "implied_team_total"]


def walk_forward_yardage(df: pd.DataFrame, test_seasons: list[int]):
    """Returns (model_pred_mean, model_resid_std, naive_pred) aligned to df.index,
    NaN outside test_seasons."""
    n = len(df)
    model_pred = np.full(n, np.nan)
    resid_std = np.full(n, np.nan)
    naive_pred = np.full(n, np.nan)

    test_mask = df["season"].isin(test_seasons)
    test_keys = df.loc[test_mask, ["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    for season, week in test_keys.itertuples(index=False):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        target_idx = df.index[(df["season"] == season) & (df["week"] == week)]
        if len(train) < 50:
            continue

        X_train = train[FEATURES].values
        y_train = train["actual"].values
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(X_train, y_train)
        train_resid_std = float(np.std(y_train - model.predict(X_train)))

        X_test = df.loc[target_idx, FEATURES].values
        preds = model.predict(X_test)
        pos = df.index.get_indexer(target_idx)
        model_pred[pos] = preds
        resid_std[pos] = max(train_resid_std, 1e-6)
        naive_pred[pos] = df.loc[target_idx, "own_trailing_avg"].values

    return model_pred, resid_std, naive_pred


def walk_forward_anytime_td(df: pd.DataFrame, test_seasons: list[int]):
    n = len(df)
    model_pred = np.full(n, np.nan)

    test_mask = df["season"].isin(test_seasons)
    test_keys = df.loc[test_mask, ["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    for season, week in test_keys.itertuples(index=False):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        target_idx = df.index[(df["season"] == season) & (df["week"] == week)]
        if len(train) < 50 or train["actual"].nunique() < 2:
            continue

        X_train = train[FEATURES].values
        y_train = train["actual"].values
        model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
        model.fit(X_train, y_train)

        X_test = df.loc[target_idx, FEATURES].values
        preds = model.predict_proba(X_test)[:, 1]
        pos = df.index.get_indexer(target_idx)
        model_pred[pos] = preds

    return model_pred


def yardage_over_prob(mean, std, line):
    return 1.0 - norm.cdf(line, loc=mean, scale=std)

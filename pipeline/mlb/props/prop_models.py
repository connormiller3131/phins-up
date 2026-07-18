"""Walk-forward models for MLB player props, same design as NFL:
- count props (hits, total bases, pitcher Ks): RidgeCV regression + Normal
  approximation for over/under a line.
- anytime HR: LogisticRegressionCV (binary), same treatment as NFL's anytime TD.
Refit periodically using only strictly earlier rows -- no leakage. MLB plays
daily rather than weekly, so refitting happens per calendar date, not per week."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

FEATURES = ["own_trailing_avg", "opp_allowed_trailing_avg"]


def walk_forward_count_stat(df: pd.DataFrame, test_dates):
    """Refits once per unique test date. test_dates: sorted array-like of
    pd.Timestamp to treat as held-out."""
    n = len(df)
    model_pred = np.full(n, np.nan)
    resid_std = np.full(n, np.nan)
    naive_pred = np.full(n, np.nan)

    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < 200:
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


def walk_forward_binary_stat(df: pd.DataFrame, test_dates):
    n = len(df)
    model_pred = np.full(n, np.nan)

    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < 200 or train["actual"].nunique() < 2:
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


def over_prob(mean, std, line):
    return 1.0 - norm.cdf(line, loc=mean, scale=std)

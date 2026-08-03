"""Walk-forward models for MLB player props, same design as NFL:
- count props (hits, total bases, pitcher Ks): RidgeCV regression + Normal
  approximation for over/under a line.
- anytime HR: LogisticRegressionCV (binary), same treatment as NFL's anytime TD.
Refit periodically using only strictly earlier rows -- no leakage. MLB plays
daily rather than weekly, so refitting happens per calendar date, not per week."""
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom
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
    """Continuous (Normal) over-probability. Kept for the historical
    backtests that recorded their numbers against it; live MLB count props
    use count_over_prob below instead -- see its docstring for the real,
    measured reason."""
    return 1.0 - norm.cdf(line, loc=mean, scale=std)


def estimate_dispersion(actual, predicted):
    """Method-of-moments negative-binomial dispersion alpha, defined by
    Var(Y|X) = mu + alpha * mu**2. alpha=0 is exactly Poisson (variance
    equals the mean); alpha>0 means the stat is over-dispersed, which real
    batting stats are -- one swing can produce 4 total bases, so total bases
    scatters much wider than its own mean would allow under Poisson.

    Solved from the training residuals: E[(y-mu)^2] = E[mu] + alpha*E[mu^2].
    Clipped at 0 because a negative estimate means under-dispersion, which
    the negative binomial cannot represent and which Poisson already covers
    as its floor case."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-6, None)
    mean_sq_resid = float(np.mean((actual - predicted) ** 2))
    mean_mu = float(np.mean(predicted))
    mean_mu_sq = float(np.mean(predicted ** 2))
    if mean_mu_sq <= 0:
        return 0.0
    return max(0.0, (mean_sq_resid - mean_mu) / mean_mu_sq)


def count_over_prob(mean, line, dispersion=0.0):
    """P(actual > line) for a DISCRETE count stat (hits, total bases, RBI,
    walks, strikeouts...), which is what these props actually are.

    Two things the Normal version above gets wrong on real MLB data, both
    confirmed against 13k+ real graded predictions:

    1. Discreteness. The site grades a prop as "over" when actual > line, so
       for an integer line of 1 the real event is "2 or more" -- there is no
       such thing as 1.4 hits. A Normal CDF evaluated at 1.0 happily counts
       the impossible mass between 1 and 2 as over. Using 1 - cdf(floor(line))
       states the real event exactly, and is correct for half-lines too
       (floor(0.5) = 0, floor(1.5) = 1).

    2. Shape. A Normal is symmetric, but a hitter's line is right-skewed with
       a large point mass at zero. Fitting one pooled residual std across all
       players and centering a symmetric bell on a mean near 0.5 puts far too
       much weight above the line.

    Together these ran the batting props about 15 points hot: Hits predicted
    48.7% over against a real 32.5%, RBI 45.7% against 26.1%. Pitcher props
    were much closer, which fits -- their counts (strikeouts ~5, outs ~15)
    are large enough for a Normal approximation to hold.

    dispersion is the alpha from estimate_dispersion; 0 gives Poisson."""
    mean = np.clip(np.asarray(mean, dtype=float), 1e-6, None)
    k = np.floor(np.asarray(line, dtype=float))
    # Below the lowest possible line every outcome clears it; guard so a
    # nonsensical negative line can't return a probability above 1.
    if dispersion is None or dispersion <= 1e-9:
        return np.clip(1.0 - poisson.cdf(k, mu=mean), 0.0, 1.0)
    n = 1.0 / dispersion
    p = 1.0 / (1.0 + dispersion * mean)
    return np.clip(1.0 - nbinom.cdf(k, n, p), 0.0, 1.0)

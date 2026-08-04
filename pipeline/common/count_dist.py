"""Turning a projected mean into P(actual > line).

Shared by MLB and NFL. Which of these is right is an empirical question per
stat, not a style choice, and both sports' backtests reached the same
conclusion from opposite directions: the answer depends on the SHAPE of the
stat, and one distribution for everything is wrong.

  count_over_prob     discrete (Poisson / negative binomial). Correct when
                      the stat is a small integer count, where the gap
                      between "1" and "2" is most of the distribution.
                      MLB: every prop. NFL: Receptions (~3/game), Passing
                      TDs (~1.8/game).

  empirical_over_prob no assumed shape at all -- reads P off the training
                      residual distribution directly. Correct when the stat
                      is large enough that discreteness is irrelevant but
                      right-skewed enough that a symmetric Normal is badly
                      centered. NFL yardage and carries.

  (Normal)            still lives in each sport's prop_models. Correct when
                      counts are large and roughly symmetric -- NFL
                      Completions (~19), Pass Attempts (~28), Passing Yards.

Real numbers behind that split are in each sport's backtest: MLB's
backtest_count_distribution.py and NFL's backtest_count_volume.py.
"""
import numpy as np
from scipy.stats import poisson, nbinom


def estimate_dispersion(actual, predicted):
    """Method-of-moments negative-binomial alpha: Var = mu + alpha*mu^2.
    Solved from the mean squared residual, so alpha > 0 means the stat is
    genuinely overdispersed relative to Poisson. Clipped at 0 (a stat can be
    underdispersed in a sample; treating that as Poisson is the honest
    fallback rather than inventing a negative variance)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-6, None)
    mean_sq_resid = float(np.mean((actual - predicted) ** 2))
    mean_mu = float(np.mean(predicted))
    mean_mu_sq = float(np.mean(predicted ** 2))
    if mean_mu_sq <= 0:
        return 0.0
    return float(max((mean_sq_resid - mean_mu) / mean_mu_sq, 0.0))


def count_over_prob(mean, line, dispersion=0.0):
    """P(actual > line) for a DISCRETE count stat.

    Two things a Normal gets wrong on real data, both confirmed against 13k+
    real graded MLB predictions:

    1. Discreteness. The site grades "over" as actual > line, so for an
       integer line of 1 the real event is "2 or more" -- there is no such
       thing as 1.4 hits. A Normal CDF at 1.0 counts the impossible mass
       between 1 and 2 as over. 1 - cdf(floor(line)) states the real event,
       and is correct for half-lines too (floor(0.5)=0, floor(1.5)=1).

    2. Shape. A Normal is symmetric, but these counts are right-skewed with a
       large point mass at zero.

    Together these ran MLB's batting props ~15 points hot (Hits 48.7%
    predicted vs 32.5% real). NFL Receptions and Passing TDs showed the same
    signature: +0.152 and +0.149 calibration gap under the Normal.

    dispersion is the alpha from estimate_dispersion; 0 gives Poisson."""
    mean = np.clip(np.asarray(mean, dtype=float), 1e-6, None)
    k = np.floor(np.asarray(line, dtype=float))
    if dispersion is None or dispersion <= 1e-9:
        return np.clip(1.0 - poisson.cdf(k, mu=mean), 0.0, 1.0)
    n = 1.0 / dispersion
    p = 1.0 / (1.0 + dispersion * mean)
    return np.clip(1.0 - nbinom.cdf(k, n, p), 0.0, 1.0)


def empirical_over_prob(mean, line, resid_sorted):
    """P(actual > line) read straight off the TRAINING residual distribution:
    the fraction of past residuals big enough to carry this row's projected
    mean over the line. Imposes no shape.

    This exists because NFL yardage is badly served by every parametric
    option tested. The Normal is symmetric, so with a line set at the
    projected mean it states ~50% -- but yardage is right-skewed, the mean
    sits above the median, and the real clear rate is ~35%. Gamma and
    lognormal fix that skew and were still clearly worse (receiving yards
    Brier 0.2523 Normal vs 0.2902 gamma): tying variance to mu^2 makes the
    distribution far too wide for high-volume players and destroys
    discrimination. The empirical residuals inherit the real skew without
    inheriting a wrong variance model, and won outright: receiving yards
    Brier 0.2523 -> 0.2341, calibration gap +0.142 -> +0.025.

    Assumes residuals are roughly homoscedastic -- the same assumption the
    single pooled residual std already made, so this is not a new one."""
    resid_sorted = np.asarray(resid_sorted, dtype=float)
    if resid_sorted.size == 0:
        return np.full(np.shape(np.asarray(line, dtype=float)), 0.5)
    needed = np.asarray(line, dtype=float) - np.asarray(mean, dtype=float)
    above = resid_sorted.size - np.searchsorted(resid_sorted, needed, side="right")
    return np.clip(above / resid_sorted.size, 0.0, 1.0)

"""Walk-forward models for player props:
- yardage props (continuous): RidgeCV regression on [own_trailing_avg, opp_allowed_trailing_avg]
- anytime-TD (binary): LogisticRegressionCV on [own_trailing_avg, opp_allowed_trailing_avg]
Both refit once per test week using only strictly earlier rows -> no leakage.
Alpha/C are chosen by internal cross-validation, not hand-picked."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

from pipeline.common.count_dist import count_over_prob, empirical_over_prob


FEATURES = ["own_trailing_avg", "opp_allowed_trailing_avg", "is_dome", "temp", "wind", "own_rest", "implied_team_total"]

# Per-stat configuration, every value chosen by walk-forward backtest on the
# untouched 2025 season (pipeline/nfl/props/backtest_count_volume.py), never
# by assumption. Two independent questions were tested per stat:
#
# "dist" -- how a projected mean becomes P(over). One distribution for all
#   eight props was wrong, and the backtest is unusually clean about why,
#   because each candidate wins only in the regime theory says it should and
#   loses badly outside it:
#     negbin    small discrete counts, where the step from 1 to 2 is most of
#               the distribution. Receptions 0.2447 -> 0.2189 Brier, gap
#               +0.152 -> +0.033. Passing TDs 0.2384 -> 0.2162, +0.149 ->
#               +0.016. (Empirical residuals do NOT work here: 0.2267 and
#               0.2275, clearly worse -- with counts this small the shape
#               genuinely is the discreteness.)
#     empirical right-skewed yardage/carries. A Normal centered on the mean
#               states ~50% at its own line, but the real clear rate is ~35%
#               because the mean sits above the median. Receiving yards
#               0.2523 -> 0.2341, gap +0.142 -> +0.025; rushing yards 0.2548
#               -> 0.2403; carries 0.2522 -> 0.2403. Gamma and lognormal
#               were tested first for exactly this and REJECTED -- they fix
#               the skew but tie variance to mu^2, which is far too wide for
#               high-volume players and cost more discrimination than the
#               calibration was worth (receiving yards 0.2902 and 0.3009).
#     normal    large, roughly symmetric counts, where nothing is broken:
#               Completions (~19/game), Pass Attempts (~28), Passing Yards.
#               Every alternative scored worse on these, and Poisson failed
#               catastrophically on yardage (0.3392), which is the control
#               confirming this split is not just noise-mining.
#
# "volume" -- an opportunity feature (own_trailing_volume). A counting stat
#   is rate x opportunities, and own_trailing_avg blends the two, so a
#   high-usage player in a slump and a low-usage player on a heater look
#   identical. Helps where usage is genuinely distinct from the stat
#   (targets vs receiving yards, carries vs rushing yards). Deliberately
#   None where the volume column IS effectively the stat -- adding attempts
#   to Completions/Pass Attempts/Passing Yards is near-collinear and
#   measurably hurt (Completions Brier 0.2405 -> 0.2413).
PROP_CONFIG = {
    "passing_yards":   {"positions": ["QB"],             "dist": "normal",    "volume": None},
    "passing_tds":     {"positions": ["QB"],             "dist": "negbin",    "volume": "attempts"},
    "completions":     {"positions": ["QB"],             "dist": "normal",    "volume": None},
    "attempts":        {"positions": ["QB"],             "dist": "normal",    "volume": None},
    "rushing_yards":   {"positions": ["RB"],             "dist": "empirical", "volume": "carries"},
    "carries":         {"positions": ["RB"],             "dist": "empirical", "volume": None},
    "receiving_yards": {"positions": ["RB", "WR", "TE"], "dist": "empirical", "volume": "targets"},
    "receptions":      {"positions": ["RB", "WR", "TE"], "dist": "negbin",    "volume": "targets"},
}


def prop_features(stat_col):
    """Feature list for one stat: the shared seven, plus the opportunity
    feature where the backtest showed it earns its place."""
    cfg = PROP_CONFIG.get(stat_col, {})
    return FEATURES + (["own_trailing_volume"] if cfg.get("volume") else [])


def prop_over_prob(prep, mean, line):
    """P(actual > line) using whichever distribution won for this stat.
    prep is the dict from prepare_count_model."""
    dist = prep.get("dist", "normal")
    if dist == "negbin":
        return float(count_over_prob(mean, line, prep.get("dispersion", 0.0)))
    if dist == "empirical":
        return float(empirical_over_prob(mean, line, prep["resid_sorted"]))
    return float(yardage_over_prob(mean, prep["resid_std"], line))


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


def walk_forward_anytime_td(df: pd.DataFrame, test_seasons: list[int], features: list[str] | None = None):
    """features defaults to the shared seven, so existing callers are
    unchanged; backtest_td_volume.py passes an extended list to test whether
    an opportunity feature earns its place here the way it does for the
    count props."""
    features = FEATURES if features is None else features
    n = len(df)
    model_pred = np.full(n, np.nan)

    test_mask = df["season"].isin(test_seasons)
    test_keys = df.loc[test_mask, ["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    for season, week in test_keys.itertuples(index=False):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        target_idx = df.index[(df["season"] == season) & (df["week"] == week)]
        if len(train) < 50 or train["actual"].nunique() < 2:
            continue

        X_train = train[features].values
        y_train = train["actual"].values
        model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
        model.fit(X_train, y_train)

        X_test = df.loc[target_idx, features].values
        preds = model.predict_proba(X_test)[:, 1]
        pos = df.index.get_indexer(target_idx)
        model_pred[pos] = preds

    return model_pred


def yardage_over_prob(mean, std, line):
    return 1.0 - norm.cdf(line, loc=mean, scale=std)

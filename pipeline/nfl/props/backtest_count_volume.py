"""Two questions for the NFL count props, tested walk-forward on the
untouched 2025 season (train strictly earlier, refit per week -- same
convention as backtest_props.py):

A. Distribution: every NFL prop probability comes from yardage_over_prob, a
   symmetric Normal with one pooled residual std. MLB just proved that
   machinery runs discrete low-count props badly hot (~15 points), and two
   NFL props sit even deeper in that regime than MLB's worst offenders:
   Receptions (mean ~3.0) and Passing TDs (mean ~1.8). Candidates: the
   deployed Normal, Poisson, negative binomial (dispersion fit on train
   only) -- reusing pipeline/mlb/props/prop_models.py's validated
   count_over_prob/estimate_dispersion, not a reimplementation. Yardage
   stats are included as controls: they're integer-valued too, so negbin is
   admissible, but their counts are large enough that the Normal should
   roughly hold -- if negbin "wins" everywhere including yards, that's a
   sign of something else going on, not discreteness.

B. Features: does adding the player's recent real usage volume
   (own_trailing_volume: targets for receiving stats, attempts for QB
   stats, carries for rushing yards) beat the deployed 7-feature set? Same
   rate-x-opportunities argument as MLB's volume hypothesis. Scored with
   each side's best distribution from part A, on a common neutral line
   (naive own-average rounded to the nearest half, identical for both), so
   Brier compares the same events.

Both parts also report RMSE (line-free projection quality).
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.props.prop_data import build_prop_table
from pipeline.nfl.props.prop_models import FEATURES, yardage_over_prob
from pipeline.mlb.props.prop_models import count_over_prob, estimate_dispersion
from pipeline.common.metrics import brier_score, log_loss
from sklearn.linear_model import RidgeCV
from scipy.stats import gamma as gamma_dist, norm

TEST_SEASONS = [2025]

# stat -> (positions, volume_col or None). Volume is skipped where the stat
# IS the volume (carries, attempts) -- a trailing-mean-of-itself feature
# would be nearly collinear with own_trailing_avg.
STATS = {
    "receptions": (["RB", "WR", "TE"], "targets"),
    "passing_tds": (["QB"], "attempts"),
    "completions": (["QB"], "attempts"),
    "attempts": (["QB"], None),
    "carries": (["RB"], None),
    "receiving_yards": (["RB", "WR", "TE"], "targets"),
    "rushing_yards": (["RB"], "carries"),
    "passing_yards": (["QB"], "attempts"),
}


def walk_forward(df, features, min_train=50):
    n = len(df)
    pred = np.full(n, np.nan)
    std = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    shape = np.full(n, np.nan)
    test_mask = df["season"].isin(TEST_SEASONS)
    test_keys = df.loc[test_mask, ["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    emp = np.full(n, np.nan)

    for season, week in test_keys.itertuples(index=False):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        target_idx = df.index[(df["season"] == season) & (df["week"] == week)]
        if len(train) < min_train or len(target_idx) == 0:
            continue
        X_train, y_train = train[features].values, train["actual"].values
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        pos = df.index.get_indexer(target_idx)
        mu_test = model.predict(df.loc[target_idx, features].values)
        pred[pos] = mu_test
        std[pos] = max(float(np.std(y_train - train_pred)), 1e-6)
        disp[pos] = estimate_dispersion(y_train, train_pred)
        shape[pos] = estimate_gamma_shape(y_train, train_pred)

        # Empirical residual candidate: impose no shape at all. Estimate
        # P(Y > line) as the fraction of TRAINING residuals large enough to
        # carry that row's own predicted mean over the line. Inherits the
        # real residual skew (which is the whole problem with the Normal
        # here) without also inheriting gamma's constant-CV variance
        # assumption, which is what wrecked gamma's discrimination.
        resid_sorted = np.sort(y_train - train_pred)
        naive_t = df.loc[target_idx, "own_trailing_avg"].values
        line_t = np.round(naive_t * 2) / 2.0
        line_t = np.where(line_t <= 0, 0.5, line_t)
        needed = line_t - mu_test
        emp[pos] = 1.0 - np.searchsorted(resid_sorted, needed, side="right") / len(resid_sorted)

    return pred, std, disp, shape, emp


def estimate_gamma_shape(actual, predicted):
    """Method-of-moments shape k for a Gamma with constant coefficient of
    variation: Var(Y|X) = mu^2 / k, so k = E[mu^2] / E[(y-mu)^2]. Constant-CV
    (multiplicative) noise is the natural fit for yardage -- a 100-yard
    receiver swings in absolute terms far more than a 20-yard one, which a
    single pooled additive std cannot express."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-6, None)
    mean_sq_resid = float(np.mean((actual - predicted) ** 2))
    mean_mu_sq = float(np.mean(predicted ** 2))
    if mean_sq_resid <= 0:
        return 50.0
    return float(np.clip(mean_mu_sq / mean_sq_resid, 0.05, 50.0))


def score(mu, sd, al, over, common_line, shape, emp):
    """Brier/gap for each distribution candidate on one prediction set.

    normal/poisson/negbin as before, plus two right-skewed CONTINUOUS
    candidates aimed at the yardage props. Those showed a large positive gap
    under the Normal (it states ~50% at a line set on the mean) because
    yardage is right-skewed: the mean sits above the median, so a line at the
    mean is cleared less than half the time. Poisson gets the skew direction
    right but is wrong-shaped for yards (it forces variance = mean, off by
    more than an order of magnitude), which is why it scored terribly there.
    Gamma and lognormal are skewed AND continuous, with variance tied to
    mu^2 rather than mu."""
    mu_pos = np.clip(mu, 1e-6, None)
    line_pos = np.clip(common_line, 1e-6, None)
    sigma_sq = np.log1p(1.0 / shape)
    sigma = np.sqrt(sigma_sq)
    out = {}
    for name, p in {
        "normal": yardage_over_prob(mu, sd, common_line),
        "poisson": count_over_prob(mu_pos, common_line, 0.0),
        "negbin": count_over_prob(mu_pos, common_line, float(np.nanmean(al))),
        "gamma": gamma_dist.sf(line_pos, a=shape, scale=mu_pos / shape),
        "lognormal": norm.sf((np.log(line_pos) - (np.log(mu_pos) - sigma_sq / 2.0)) / sigma),
        "empirical": emp,
    }.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out[name] = {"brier": brier_score(over, p), "log_loss": log_loss(over, p),
                     "gap": float(p.mean()) - float(over.mean())}
    return out


def main():
    print("=== NFL count-prop distribution + volume backtest ===")
    print(f"Walk-forward, untouched test season(s) {TEST_SEASONS}\n")
    results = {}

    for stat, (positions, volume_col) in STATS.items():
        df = build_prop_table(stat, positions, volume_col=volume_col)
        base_feats = FEATURES
        vol_feats = FEATURES + ["own_trailing_volume"] if volume_col else None

        runs = {"baseline": walk_forward(df, base_feats)}
        if vol_feats:
            runs["+volume"] = walk_forward(df, vol_feats)

        valid = df["season"].isin(TEST_SEASONS).values
        for pred, _, _, _, _ in runs.values():
            valid &= ~np.isnan(pred)
        if valid.sum() < 100:
            print(f"--- {stat}: only {int(valid.sum())} usable rows, skipped ---")
            continue

        actual = df["actual"].values[valid]
        naive = df["own_trailing_avg"].values[valid]
        common_line = np.round(naive * 2) / 2.0
        common_line = np.where(common_line <= 0, 0.5, common_line)
        over = (actual > common_line).astype(float)

        print(f"--- {stat} (n={int(valid.sum())}, over-rate {over.mean():.3f}) ---")
        stat_out = {"n": int(valid.sum()), "actual_rate": float(over.mean()), "feature_sets": {}}
        for fname, (pred, std, disp, shp, emp) in runs.items():
            mu, sd, al = pred[valid], std[valid], disp[valid]
            rmse = float(np.sqrt(np.mean((actual - mu) ** 2)))
            dists = score(mu, sd, al, over, common_line, float(np.nanmean(shp[valid])), emp[valid])
            stat_out["feature_sets"][fname] = {"rmse": rmse, "distributions": dists}
            for dname, r in dists.items():
                print(f"    {fname:<10} {dname:<8} rmse={rmse:>8.3f} brier={r['brier']:.4f} "
                      f"logloss={r['log_loss']:.4f} gap={r['gap']:+.3f}")

        results[stat] = stat_out
        print()

    print("=== summary ===")
    for stat, r in results.items():
        base = r["feature_sets"]["baseline"]
        best_dist = min(base["distributions"], key=lambda k: base["distributions"][k]["brier"])
        line = (f"  {stat:<18} best dist: {best_dist:<7} "
                f"(normal brier {base['distributions']['normal']['brier']:.4f} -> "
                f"{base['distributions'][best_dist]['brier']:.4f}, "
                f"gap {base['distributions']['normal']['gap']:+.3f} -> "
                f"{base['distributions'][best_dist]['gap']:+.3f})")
        if "+volume" in r["feature_sets"]:
            vol = r["feature_sets"]["+volume"]
            d_rmse = base["rmse"] - vol["rmse"]
            vb = min(vol["distributions"].values(), key=lambda x: x["brier"])["brier"]
            bb = min(base["distributions"].values(), key=lambda x: x["brier"])["brier"]
            line += f" | volume: rmse {'-' if d_rmse >= 0 else '+'}{abs(d_rmse):.3f}, brier {bb:.4f} -> {vb:.4f}"
        print(line)

    path = ROOT / "notebooks_out" / "nfl_count_volume_backtest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

"""Does adding a career-to-date (expanding-window) trailing HR rate fix the
Anytime HR model's floor-probability problem for players with essentially no
power history?

Real, reported case: Chandler Simpson (1 career HR across pro ball) still
showed a ~7.7% Anytime HR probability. Traced to the model's own intercept:
with own_trailing_avg pinned at its floor of 0.0 (can't go negative), the
2-feature [own_trailing_avg, opp_allowed_trailing_avg] LogisticRegressionCV
has no way to distinguish "0 HR in the last 15 games, but plenty of power
before that" from "structurally near-zero power" -- both hit the same floor,
and the intercept + opponent term alone still produce a real, non-trivial
probability (confirmed: own_trailing_avg=0.0 against a real range of
opponent allowed-rates predicts 7.6%-8.3% under the current shipped model).

own_career_trailing_avg (an expanding, not window-capped, shift(1) mean --
see prop_data.py) gives the model a second, independent read on the same
player: for a rookie it's ~identical to own_trailing_avg (no extra history
yet); for a veteran with a real 0-power track record, it stays near 0 even
when the 15-game window also happens to be near 0, letting the model learn
that stacked near-zero evidence should predict much lower than either signal
alone. Same rigor as every other prop backtest in this project: real
historical data, walk-forward (no leakage), 2025-08-01 to 2025-09-30 test
window, Brier score + bootstrap significance test before shipping.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.props.prop_data import build_batter_prop_table
from pipeline.common.metrics import brier_score, log_loss, accuracy

TEST_START = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2025-09-30")

BASELINE_FEATURES = ["own_trailing_avg", "opp_allowed_trailing_avg"]
CAREER_FEATURES = ["own_trailing_avg", "own_career_trailing_avg", "opp_allowed_trailing_avg"]


def bootstrap_brier_diff(y_test, preds_a, preds_b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    diffs = np.empty(n_boot)
    y_test, preds_a, preds_b = np.asarray(y_test), np.asarray(preds_a), np.asarray(preds_b)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = brier_score(y_test[idx], preds_a[idx]) - brier_score(y_test[idx], preds_b[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def walk_forward_binary(df, features, test_dates):
    n = len(df)
    preds = np.full(n, np.nan)
    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < 200 or train["actual"].nunique() < 2:
            continue
        model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
        model.fit(train[features].values, train["actual"].values)
        pos = df.index.get_indexer(target_idx)
        preds[pos] = model.predict_proba(df.loc[target_idx, features].values)[:, 1]
    return preds


def main():
    df = build_batter_prop_table("home_runs")
    df["actual"] = (df["actual"] > 0).astype(float)
    df = df.sort_values("game_date").reset_index(drop=True)

    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    print(f"Rows: {len(df)}. Test window {TEST_START.date()} to {TEST_END.date()}: {len(test_dates)} dates.")

    base_preds = walk_forward_binary(df, BASELINE_FEATURES, test_dates)
    career_preds = walk_forward_binary(df, CAREER_FEATURES, test_dates)

    test_mask = df["game_date"].between(TEST_START, TEST_END)
    valid = test_mask.values & ~np.isnan(base_preds) & ~np.isnan(career_preds)
    y = df["actual"].values[valid]
    base_p = base_preds[valid]
    career_p = career_preds[valid]

    print(f"\nEvaluated on {valid.sum()} held-out rows.")
    print(f"\nbaseline (own+opp):")
    print(f"  Brier:    {brier_score(y, base_p):.5f}")
    print(f"  Log loss: {log_loss(y, base_p):.5f}")
    print(f"  Accuracy: {accuracy(y, base_p):.5f}")

    print(f"\n+career (own+career+opp):")
    print(f"  Brier:    {brier_score(y, career_p):.5f}")
    print(f"  Log loss: {log_loss(y, career_p):.5f}")
    print(f"  Accuracy: {accuracy(y, career_p):.5f}")

    ci_lo, ci_hi = bootstrap_brier_diff(y, career_p, base_p)
    print(f"\nBootstrap 95% CI on (career_brier - baseline_brier): [{ci_lo:.5f}, {ci_hi:.5f}]")
    improves = ci_hi < 0
    print(f"  {'IMPROVES' if improves else 'DOES NOT reliably improve'} on baseline at 95% confidence.")

    # The actual reported symptom: rows where the 15-game window is exactly
    # 0 (own_trailing_avg == 0) -- does adding the career signal separate a
    # real zero-power veteran from a rookie/short-slump case that just
    # happens to also read 0 right now?
    zero_recent = df[valid].assign(base_p=base_p, career_p=career_p)
    zero_recent = zero_recent[zero_recent["own_trailing_avg"] == 0.0]
    print(f"\nRows with own_trailing_avg==0.0 in test set: {len(zero_recent)}")
    if len(zero_recent):
        low_career = zero_recent[zero_recent["own_career_trailing_avg"] < 0.02]
        high_career = zero_recent[zero_recent["own_career_trailing_avg"] >= 0.02]
        print(f"  of those, {len(low_career)} also have a near-zero career rate (<0.02/game, i.e. a real Simpson-like case):")
        if len(low_career):
            print(f"    mean baseline pred: {low_career['base_p'].mean():.4f}   mean +career pred: {low_career['career_p'].mean():.4f}")
            print(f"    actual HR rate in this group: {low_career['actual'].mean():.4f}")
        print(f"  the remaining {len(high_career)} have real power history despite a current cold streak:")
        if len(high_career):
            print(f"    mean baseline pred: {high_career['base_p'].mean():.4f}   mean +career pred: {high_career['career_p'].mean():.4f}")
            print(f"    actual HR rate in this group: {high_career['actual'].mean():.4f}")


if __name__ == "__main__":
    main()

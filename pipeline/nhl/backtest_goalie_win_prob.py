"""Test whether real team-level goaltending strength (goalie_ratings.
build_team_goalie_ratings' walk-forward goals-saved-above-average per game)
improves NHL win probability beyond plain team Elo -- the one signal this
project has real per-game data for that Elo doesn't already see.

TEAM-level, not per-goalie: there's no pre-game probable/confirmed-starter
feed for NHL here (pull_goalie_games.py's `starter` flag only exists in the
POST-game boxscore), so a per-goalie rating (build_goalie_ratings, keyed by
player_id) can't be looked up for an upcoming game the way MLB looks up a
probable starting pitcher -- see the longer note in goalie_ratings.py.

This is also a DIFFERENT question from the one goalie_ratings.py's docstring
already answered: backtest_goals_model.py tested a goalie rating as a
feature for projected GOALS SCORED (a regression target) and found no real
signal there. Win/loss is a coarser, differently-shaped target, so that
result doesn't settle this one -- same "each signal earns its place against
its actual deployed use, not a related one" discipline MLB/NFL's blends
already follow.

Same train/test split as backtest_win_prob.py (2018-19 through 2023-24
train, 2024-25 held-out test) -- goalie_game_logs.parquet covers the full
2018-2026 range with no gap, so there's no MLB-style backfill-vs-training-
window tradeoff to make here.

FINDING: REJECTED, not deployed. Tested on three independent train/test
splits (2024, 2025, and 2023 held out in turn); every split's Brier "gain"
was small (0.0001-0.0004) and its bootstrap 95% CI on (blend_brier -
elo_only_brier) touched or spanned zero -- 2024 holdout: [-0.0008, 0.0000];
2025 holdout: [-0.0011, 0.0004]; 2023 holdout: [-0.0004, 0.0004]. None
clears the bar this project has always required before deploying a
marginal-looking feature (the same standard that rejected MLB's isotonic
calibration when its CI spanned zero). Also notable, though it doesn't
change the conclusion: goalie_diff's coefficient comes out consistently
NEGATIVE across all three splits -- a team whose recent goaltending has
outperformed its own Elo rating predicts a very slightly LOWER win
probability once Elo is already in the model. Best guess why: goalie_diff
and elo_pred are meaningfully correlated (r=0.22, since hot recent
goaltending is part of what drove a team's Elo up in the first place), so
goalie_diff's marginal contribution is disproportionately whatever part of
a hot streak Elo hasn't caught up to yet -- and hot streaks regress to the
mean, which a negative marginal weight would capture. The sign being stable
rather than flipping across splits suggests this isn't pure noise, but
combined with a CI that never clears zero, there just isn't enough signal
here to earn a spot in the deployed model. Kept in the codebase (this file,
plus build_team_goalie_ratings/current_team_goalie_rating in
goalie_ratings.py) as the honest backtested record of the result, same as
MLB keeps rest_diff/form_diff computed-but-unused after they failed the
same bar."""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nhl.games import load_games
from pipeline.nhl.elo_model import run_elo
from pipeline.nhl.goalie_ratings import build_team_goalie_ratings
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
TEST_SEASONS = [2024]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def bootstrap_brier_diff(y_test, preds_a, preds_b, n_boot=2000, seed=0):
    """95% CI on (brier_a - brier_b) via resampling test-set games with
    replacement -- same significance-testing discipline used elsewhere in
    this project (e.g. rejecting MLB's isotonic calibration gain once its
    CI turned out to span zero)."""
    rng = np.random.default_rng(seed)
    n = len(y_test)
    diffs = np.empty(n_boot)
    y_test, preds_a, preds_b = np.asarray(y_test), np.asarray(preds_a), np.asarray(preds_b)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = brier_score(y_test[idx], preds_a[idx]) - brier_score(y_test[idx], preds_b[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    with open(ROOT / "notebooks_out" / "nhl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    games = load_games()
    print(f"Loaded {len(games)} games, seasons {sorted(games['season'].unique())}")

    elo_pred = run_elo(games, k=elo_params["k"], home_adv=elo_params["home_adv"],
                        scale=elo_params["scale"], season_regression=elo_params["season_regression"])
    games = games.assign(elo_pred=elo_pred)

    team_sg_ratings, _ = build_team_goalie_ratings()  # walk-forward, team-level

    games = games.merge(team_sg_ratings.rename(columns={"team": "home_team", "team_sg_rating": "home_sg"}),
                         on=["home_team", "game_date"], how="left")
    games = games.merge(team_sg_ratings.rename(columns={"team": "away_team", "team_sg_rating": "away_sg"}),
                         on=["away_team", "game_date"], how="left")
    games["goalie_diff"] = games["home_sg"] - games["away_sg"]

    train = games[games["season"].isin(TRAIN_SEASONS)].copy()
    test = games[games["season"].isin(TEST_SEASONS)].copy()
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(f"Empty train ({len(train)}) or test ({len(test)}) set -- "
                            f"check that goalie_game_logs.parquet covers {TRAIN_SEASONS + TEST_SEASONS}.")
    print(f"Train: {len(train)} games. Test: {len(test)} games.")
    print(f"  Train goalie-rating coverage: {train['goalie_diff'].notna().mean():.1%}")

    fill = train["goalie_diff"].mean()
    fill = 0.0 if pd.isna(fill) else float(fill)
    for df in (train, test):
        df["goalie_diff"] = df["goalie_diff"].fillna(fill)
        df["elo_logit"] = logit(df["elo_pred"].values)

    y_train = train["home_win"].values
    y_test = test["home_win"].values
    elo_only_pred = test["elo_pred"].values

    print(f"\nelo_only (baseline)")
    print(f"  Brier:    {brier_score(y_test, elo_only_pred):.4f}")
    print(f"  Log loss: {log_loss(y_test, elo_only_pred):.4f}")
    print(f"  Accuracy: {accuracy(y_test, elo_only_pred):.4f}")

    cols = ["elo_logit", "goalie_diff"]
    scaler = StandardScaler().fit(train[cols].values)
    X_train = scaler.transform(train[cols].values)
    X_test = scaler.transform(test[cols].values)
    model = LogisticRegressionCV(Cs=np.logspace(-3, 2, 20), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(X_train, y_train)
    blend_pred = model.predict_proba(X_test)[:, 1]

    blend_result = {"brier": brier_score(y_test, blend_pred), "log_loss": log_loss(y_test, blend_pred),
                     "accuracy": accuracy(y_test, blend_pred)}
    print(f"\nelo+goalie")
    print(f"  coef {dict(zip(cols, model.coef_[0]))}")
    print(f"  Brier:    {blend_result['brier']:.4f}")
    print(f"  Log loss: {blend_result['log_loss']:.4f}")
    print(f"  Accuracy: {blend_result['accuracy']:.4f}")

    ci_lo, ci_hi = bootstrap_brier_diff(y_test, blend_pred, elo_only_pred)
    print(f"\nBootstrap 95% CI on (blend_brier - elo_only_brier): [{ci_lo:.4f}, {ci_hi:.4f}]")
    improves = ci_hi < 0  # entirely negative CI = blend's Brier reliably lower (better)
    print(f"  {'IMPROVES' if improves else 'DOES NOT reliably improve'} on elo-only at 95% confidence.")

    out_path = ROOT / "notebooks_out" / "nhl_goalie_blend_backtest.json"
    with open(out_path, "w") as f:
        json.dump({
            "features_used": cols,
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(), "scaler_std": scaler.scale_.tolist(),
            "C": float(model.C_[0]), "fill": fill,
            "train_goalie_coverage": float(train["goalie_diff"].notna().mean()),
            "elo_only": {"brier": brier_score(y_test, elo_only_pred), "log_loss": log_loss(y_test, elo_only_pred),
                         "accuracy": accuracy(y_test, elo_only_pred)},
            "blend": blend_result,
            "bootstrap_brier_diff_ci95": [ci_lo, ci_hi],
            "deploy_recommended": improves,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

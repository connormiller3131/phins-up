"""How much is real injury information actually worth on top of Elo?

This exists to answer a blend-weight question with data instead of taste.
The temptation is to pick a split by feel ("60/40 Elo vs injuries") but this
project fits every weight it uses -- Elo's own hyperparameters were grid-
searched, MLB's starting-pitcher blend was a logistic regression validated
out-of-sample, and the season simulator's shrinkage was swept. A hand-picked
blend would be the one number on the site nobody had checked.

The question is also subtler than "do injuries matter". Of course they do.
The question is how much they matter GIVEN Elo, and Elo already absorbs
roster quality indirectly: a team with a good quarterback wins games, which
is exactly what raises its rating. So the only thing an injury term can add
is the delta from that team's normal state -- and only for the current week,
which the rating has no way to know about.

Method: build a per-team-per-game injury summary, then compare the deployed
Elo against Elo plus injury terms in a logistic regression, walk-forward
(fit 2019-2024, test on the untouched 2025 season -- the same holdout the
Elo model itself was validated on). Reports the fitted coefficients so the
"split" is read off real data rather than assumed.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.elo_model import run_elo
from pipeline.common.metrics import brier_score, log_loss, accuracy
from sklearn.linear_model import LogisticRegression

DATA_DIR = ROOT / "data" / "nfl"
TRAIN_SEASONS = list(range(2019, 2025))
TEST_SEASONS = [2025]

# Only these count as real absences. Questionable is mostly noise -- the
# large majority of players listed Questionable play, so treating it as an
# absence would inject far more error than signal. Doubtful is closer to Out
# in practice but rarer; kept separate so the fit can price it itself.
HARD_OUT = {"Out"}
SOFT_OUT = {"Doubtful"}

# Skill positions whose absence plausibly moves a team's scoring. Offensive
# line and defensive absences are real but far harder to attribute from a
# report alone, so they are deliberately excluded rather than lumped in as
# undifferentiated noise.
SKILL = {"RB", "WR", "TE"}


def load_games():
    raw = pl.read_parquet(DATA_DIR / "schedules.parquet")
    reg = raw.filter((pl.col("game_type") == "REG") & pl.col("home_score").is_not_null())
    df = reg.select(["season", "week", "home_team", "away_team", "home_score", "away_score",
                     "location", "home_rest", "away_rest"]).to_pandas()
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df.sort_values(["season", "week"]).reset_index(drop=True)


def injury_summary():
    """Per (season, week, team): how many real absences, split by importance.
    QB is kept as its own term because its impact is not remotely comparable
    to any other position -- lumping a starting quarterback in with a backup
    safety as '2 players out' is exactly the averaging that would make an
    injury feature useless."""
    inj = pl.read_parquet(DATA_DIR / "injuries.parquet").to_pandas()
    inj = inj[inj["report_status"].notna()].copy()
    inj["hard"] = inj["report_status"].isin(HARD_OUT).astype(float)
    inj["soft"] = inj["report_status"].isin(SOFT_OUT).astype(float)
    inj["is_qb"] = (inj["position"] == "QB").astype(float)
    inj["is_skill"] = inj["position"].isin(SKILL).astype(float)

    inj["qb_out"] = inj["hard"] * inj["is_qb"]
    inj["skill_out"] = inj["hard"] * inj["is_skill"]
    inj["any_out"] = inj["hard"]
    inj["qb_doubtful"] = inj["soft"] * inj["is_qb"]

    g = inj.groupby(["season", "week", "team"], as_index=False)[
        ["qb_out", "skill_out", "any_out", "qb_doubtful"]].sum()
    g["season"] = g["season"].astype(int)
    g["week"] = g["week"].astype(int)
    return g


def attach(df, inj):
    for side in ("home", "away"):
        m = inj.rename(columns={"team": f"{side}_team", **{
            c: f"{side}_{c}" for c in ["qb_out", "skill_out", "any_out", "qb_doubtful"]}})
        df = df.merge(m, on=["season", "week", f"{side}_team"], how="left")
    cols = [f"{s}_{c}" for s in ("home", "away")
            for c in ("qb_out", "skill_out", "any_out", "qb_doubtful")]
    df[cols] = df[cols].fillna(0.0)
    # Differentials: what matters is the IMBALANCE, not the raw counts. A
    # game where both teams are missing their quarterback is not the same as
    # one where only the home team is.
    df["qb_out_diff"] = df["home_qb_out"] - df["away_qb_out"]
    df["skill_out_diff"] = df["home_skill_out"] - df["away_skill_out"]
    df["any_out_diff"] = df["home_any_out"] - df["away_any_out"]
    return df


def main():
    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    games = load_games()
    games["elo_pred"] = run_elo(games, k=elo_params["k"], home_adv=elo_params["home_adv"],
                                scale=elo_params["scale"], rest_adv=elo_params.get("rest_adv", 0.0),
                                season_regression=elo_params.get("season_regression", 0.75))
    games = attach(games, injury_summary())

    train = games[games["season"].isin(TRAIN_SEASONS)].copy()
    test = games[games["season"].isin(TEST_SEASONS)].copy()
    print(f"Train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}: {len(train)} games | Test {TEST_SEASONS}: {len(test)} games\n")

    # --- 1. Raw effect size, before any model: does a QB-out game actually
    # deviate from what Elo expected? This is the honest first check -- if
    # the residual is ~0 there is nothing for a blend to capture.
    print("=== Real effect size: actual result vs. what Elo already expected ===")
    for label, mask in [
        ("home QB out, away QB in", (games["qb_out_diff"] > 0)),
        ("away QB out, home QB in", (games["qb_out_diff"] < 0)),
        ("neither QB out",          (games["qb_out_diff"] == 0)),
    ]:
        sub = games[mask]
        if len(sub) < 20:
            print(f"  {label:<26} n={len(sub)} (too few)")
            continue
        resid = sub["home_win"].mean() - sub["elo_pred"].mean()
        print(f"  {label:<26} n={len(sub):<5} actual home win {sub['home_win'].mean():.3f} "
              f"vs Elo {sub['elo_pred'].mean():.3f}   residual {resid:+.3f}")

    # --- 2. Does a fitted blend beat Elo alone out-of-sample?
    feature_sets = {
        "elo only":            ["elo_pred"],
        "elo + QB out":        ["elo_pred", "qb_out_diff"],
        "elo + QB + skill":    ["elo_pred", "qb_out_diff", "skill_out_diff"],
        "elo + all-out count": ["elo_pred", "any_out_diff"],
    }

    print("\n=== Out-of-sample on 2025 (untouched) ===")
    print(f"  {'model':<22}{'brier':>9}{'logloss':>10}{'acc':>8}")
    results = {}
    for name, feats in feature_sets.items():
        model = LogisticRegression(max_iter=1000)
        model.fit(train[feats].values, train["home_win"].values)
        p = model.predict_proba(test[feats].values)[:, 1]
        results[name] = {
            "brier": brier_score(test["home_win"].values, p),
            "log_loss": log_loss(test["home_win"].values, p),
            "accuracy": accuracy(test["home_win"].values, p),
            "coefs": dict(zip(feats, model.coef_[0].round(4).tolist())),
        }
        r = results[name]
        print(f"  {name:<22}{r['brier']:>9.4f}{r['log_loss']:>10.4f}{r['accuracy']:>8.4f}")

    # Raw deployed Elo as the true baseline (no refit at all).
    p_raw = test["elo_pred"].values
    print(f"  {'deployed elo (raw)':<22}{brier_score(test['home_win'].values, p_raw):>9.4f}"
          f"{log_loss(test['home_win'].values, p_raw):>10.4f}"
          f"{accuracy(test['home_win'].values, p_raw):>8.4f}")

    print("\n=== Fitted coefficients (this is the 'split', read off the data) ===")
    for name, r in results.items():
        if len(r["coefs"]) > 1:
            print(f"  {name:<22}{r['coefs']}")

    # --- 3. The decisive test. A single-season aggregate is dominated by the
    # ~80% of games where no quarterback is out, so a real effect on the
    # affected minority barely moves it. Pool a walk-forward across several
    # test seasons (train on everything strictly earlier each time), then
    # score BOTH overall and on the subset where the feature is actually
    # non-zero -- that subset is the only place an injury term can do work.
    print("\n=== Walk-forward pooled over 2021-2025 (train on all prior seasons) ===")
    pooled = {name: [] for name in feature_sets}
    pooled_y, pooled_qbdiff, pooled_raw = [], [], []
    for test_season in range(2021, 2026):
        tr = games[games["season"] < test_season]
        te = games[games["season"] == test_season]
        if len(tr) < 200 or len(te) == 0:
            continue
        for name, feats in feature_sets.items():
            m = LogisticRegression(max_iter=1000)
            m.fit(tr[feats].values, tr["home_win"].values)
            pooled[name].append(m.predict_proba(te[feats].values)[:, 1])
        pooled_y.append(te["home_win"].values)
        pooled_qbdiff.append(te["qb_out_diff"].values)
        pooled_raw.append(te["elo_pred"].values)

    y = np.concatenate(pooled_y)
    qbd = np.concatenate(pooled_qbdiff)
    raw = np.concatenate(pooled_raw)
    preds = {name: np.concatenate(v) for name, v in pooled.items()}
    preds["deployed elo (raw)"] = raw
    affected = qbd != 0

    print(f"  pooled games: {len(y)}  |  with a QB-out imbalance: {int(affected.sum())}\n")
    print(f"  {'model':<22}{'brier(all)':>12}{'brier(QB-out)':>15}{'acc(QB-out)':>13}")
    pooled_out = {}
    for name, p in preds.items():
        b_all = brier_score(y, p)
        b_sub = brier_score(y[affected], p[affected])
        a_sub = accuracy(y[affected], p[affected])
        pooled_out[name] = {"brier_all": b_all, "brier_qb_out": b_sub, "acc_qb_out": a_sub}
        print(f"  {name:<22}{b_all:>12.4f}{b_sub:>15.4f}{a_sub:>13.4f}")

    out = ROOT / "notebooks_out" / "nfl_injury_effect_backtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"train_seasons": TRAIN_SEASONS, "test_seasons": TEST_SEASONS,
                   "results": results, "walk_forward_pooled": pooled_out}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

"""Does a STYLE MISMATCH improve NFL win probability, where raw unit quality
did not?

pipeline/nfl/backtest_offense_defense_model.py already tested team offense
and defense quality and rejected all seven candidates -- every one lost to
Elo alone, and its DEPLOYED list is empty. That result is not being
re-litigated here. The likely reason it failed is that those features were
MAIN EFFECTS and largely redundant with the rating: a team with a good
offense wins games, and winning games is precisely what raises its Elo.

    rush_off_diff = home_rush_ypc_off - away_rush_ypc_off   (offense v offense)
    rush_def_diff = away_rush_ypc_def - home_rush_ypc_def   (defense v defense)

What was never tested is the CROSS term -- this offense against THAT
defense. "A run-heavy team facing a bad run defense" is not a statement
about either team's overall quality; it is a statement about the pairing,
and a single team rating has no way to represent it. Two teams with
identical Elo can present completely different problems depending on who
they are playing.

Candidates here, all genuinely new:

  mismatch     home rush offense vs away rush DEFENSE (and the pass
               equivalent), differenced between the two sides. The direct
               expression of "my strength vs your weakness".
  tendency     run rate, so the model can tell a team that actually runs
               the ball from one that merely could. A rushing mismatch is
               worth more to a team that runs 55% of the time than to one
               that runs 35%.
  weighted     mismatch scaled by tendency -- the full version of the idea:
               how lopsided the pairing is, times how often the offense
               actually attacks that way.

Same discipline as every other model change here: fit on 2019-2024, report
on the untouched 2025 season, plus a pooled walk-forward for a larger test
sample, and ship nothing that does not beat the Elo-only baseline.
"""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.games import load_games
from pipeline.nfl.elo_model import run_elo
from pipeline.nfl.team_offense_defense import build_offense_defense_ratings
from pipeline.common.metrics import brier_score, log_loss, accuracy
from sklearn.linear_model import LogisticRegression

DATA_DIR = ROOT / "data" / "nfl"
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_SEASONS = [2025]
WINDOW, MIN_GAMES = 8, 3


def build_tendency():
    """Trailing run rate per team: carries / (carries + pass attempts), using
    only prior games (shift(1)), so it is walk-forward safe like every other
    trailing feature in this project."""
    ts = pl.read_parquet(DATA_DIR / "team_stats.parquet").to_pandas()
    keep = ["season", "week", "team", "carries", "attempts"]
    ts = ts[[c for c in keep if c in ts.columns]].copy()
    ts = ts.dropna(subset=["carries", "attempts"])
    ts["run_rate_game"] = ts["carries"] / (ts["carries"] + ts["attempts"]).replace(0, np.nan)
    ts = ts.sort_values(["team", "season", "week"])
    ts["run_rate"] = ts.groupby("team")["run_rate_game"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
    return ts[["season", "week", "team", "run_rate"]]


def main():
    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    games = load_games()
    games["elo_pred"] = run_elo(games, k=elo_params["k"], home_adv=elo_params["home_adv"],
                                scale=elo_params["scale"], rest_adv=elo_params.get("rest_adv", 0.0),
                                season_regression=elo_params.get("season_regression", 0.75))

    ratings = build_offense_defense_ratings()
    rating_cols = ["pass_ypa_off_trail", "rush_ypc_off_trail",
                   "pass_ypa_def_trail", "rush_ypc_def_trail"]
    have = [c for c in rating_cols if c in ratings.columns]
    short = {c: c.replace("_trail", "") for c in have}

    for side in ("home", "away"):
        r = ratings[["team", "season", "week"] + have].rename(
            columns={"team": f"{side}_team", **{c: f"{side}_{short[c]}" for c in have}})
        games = games.merge(r, on=[f"{side}_team", "season", "week"], how="left")

    tend = build_tendency()
    for side in ("home", "away"):
        t = tend.rename(columns={"team": f"{side}_team", "run_rate": f"{side}_run_rate"})
        games = games.merge(t, on=[f"{side}_team", "season", "week"], how="left")

    # THE NEW IDEA: offense against the DEFENSE it actually faces.
    games["home_rush_edge"] = games["home_rush_ypc_off"] - games["away_rush_ypc_def"]
    games["away_rush_edge"] = games["away_rush_ypc_off"] - games["home_rush_ypc_def"]
    games["home_pass_edge"] = games["home_pass_ypa_off"] - games["away_pass_ypa_def"]
    games["away_pass_edge"] = games["away_pass_ypa_off"] - games["home_pass_ypa_def"]
    games["rush_mismatch"] = games["home_rush_edge"] - games["away_rush_edge"]
    games["pass_mismatch"] = games["home_pass_edge"] - games["away_pass_edge"]

    # Tendency-weighted: a rushing edge matters more to a team that runs.
    games["rush_mismatch_w"] = (games["home_rush_edge"] * games["home_run_rate"]
                                - games["away_rush_edge"] * games["away_run_rate"])
    games["pass_mismatch_w"] = (games["home_pass_edge"] * (1 - games["home_run_rate"])
                                - games["away_pass_edge"] * (1 - games["away_run_rate"]))
    games["run_rate_diff"] = games["home_run_rate"] - games["away_run_rate"]

    feat_all = ["rush_mismatch", "pass_mismatch", "rush_mismatch_w", "pass_mismatch_w", "run_rate_diff"]
    fills = {c: float(games[c].median()) for c in feat_all}
    games[feat_all] = games[feat_all].fillna(pd.Series(fills))
    games = games.dropna(subset=["elo_pred"]).reset_index(drop=True)
    # Ties are encoded as home_win == 0.5 and are not a binary outcome; the
    # same exclusion MLB's backtests apply. Rare in the NFL but real.
    n_ties = int((games["home_win"] == 0.5).sum())
    games = games[games["home_win"] != 0.5].reset_index(drop=True)
    if n_ties:
        print(f"Excluded {n_ties} tie game(s) -- not a binary outcome.\n")

    feature_sets = {
        "elo only":                    ["elo_pred"],
        "elo + rush mismatch":         ["elo_pred", "rush_mismatch"],
        "elo + pass mismatch":         ["elo_pred", "pass_mismatch"],
        "elo + both mismatches":       ["elo_pred", "rush_mismatch", "pass_mismatch"],
        "elo + tendency-weighted":     ["elo_pred", "rush_mismatch_w", "pass_mismatch_w"],
        "elo + mismatch + run rate":   ["elo_pred", "rush_mismatch", "pass_mismatch", "run_rate_diff"],
    }

    train = games[games["season"].isin(TRAIN_SEASONS)]
    test = games[games["season"].isin(TEST_SEASONS)]
    print(f"Train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}: {len(train)} | Test {TEST_SEASONS}: {len(test)}\n")

    print("=== Held-out 2025 ===")
    print(f"  {'model':<30}{'brier':>9}{'logloss':>10}{'acc':>8}")
    base_brier = brier_score(test["home_win"].values, test["elo_pred"].values)
    results = {}
    for name, feats in feature_sets.items():
        m = LogisticRegression(max_iter=1000)
        m.fit(train[feats].values, train["home_win"].values)
        p = m.predict_proba(test[feats].values)[:, 1]
        results[name] = {"brier": brier_score(test["home_win"].values, p),
                         "log_loss": log_loss(test["home_win"].values, p),
                         "accuracy": accuracy(test["home_win"].values, p),
                         "coefs": dict(zip(feats, m.coef_[0].round(4).tolist()))}
        r = results[name]
        print(f"  {name:<30}{r['brier']:>9.4f}{r['log_loss']:>10.4f}{r['accuracy']:>8.4f}")
    print(f"  {'deployed elo (raw)':<30}{base_brier:>9.4f}")

    # Pooled walk-forward for a bigger test sample.
    print("\n=== Walk-forward pooled 2021-2025 ===")
    pooled = {n: [] for n in feature_sets}
    ys, raws = [], []
    for ts_season in range(2021, 2026):
        tr = games[games["season"] < ts_season]
        te = games[games["season"] == ts_season]
        if len(tr) < 200 or len(te) == 0:
            continue
        for name, feats in feature_sets.items():
            m = LogisticRegression(max_iter=1000)
            m.fit(tr[feats].values, tr["home_win"].values)
            pooled[name].append(m.predict_proba(te[feats].values)[:, 1])
        ys.append(te["home_win"].values)
        raws.append(te["elo_pred"].values)
    y = np.concatenate(ys)
    print(f"  pooled games: {len(y)}")
    print(f"  {'model':<30}{'brier':>9}{'logloss':>10}")
    pooled_out = {}
    for name, v in pooled.items():
        p = np.concatenate(v)
        pooled_out[name] = {"brier": brier_score(y, p), "log_loss": log_loss(y, p)}
        print(f"  {name:<30}{pooled_out[name]['brier']:>9.4f}{pooled_out[name]['log_loss']:>10.4f}")
    raw = np.concatenate(raws)
    print(f"  {'deployed elo (raw)':<30}{brier_score(y, raw):>9.4f}{log_loss(y, raw):>10.4f}")

    print("\n=== Coefficients ===")
    for name, r in results.items():
        if len(r["coefs"]) > 1:
            print(f"  {name:<30}{r['coefs']}")

    out = ROOT / "notebooks_out" / "nfl_style_matchup_backtest.json"
    with open(out, "w") as f:
        json.dump({"held_out": results, "walk_forward_pooled": pooled_out, "fills": fills}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

"""Do the model's disagreements with the betting market actually pay?

The site presents moneyline "edge" as the gap between the model's win
probability and the market's no-vig price, and the suggested parlay's value
leg, the good-value flags and the "biggest disagreement" posts all rest on the
idea that a big gap is an opportunity. This tests that idea directly: bet one
unit on whichever side the model leans to, at the market's own FAIR (no-vig)
odds, whenever the two disagree, and see whether it makes money.

Walk-forward 2021-2026, with the quarterback shift (pipeline/nfl/qb_adjust.py)
refitted on prior seasons only, so both models are what would have been live.

RESULT, 2026-09-26, adjusted model, fair odds, before any vig:

    disagreement   bets    won   market expected   fair ROI   +/- 1 s.e.
      5-10 pts      447   40.3%       41.7%          -4.9%       6.2%
     10-15 pts      235   37.4%       37.4%          +0.4%       9.5%
     15-20 pts       84   32.1%       34.4%          -4.0%      16.3%
       20+ pts       47   29.8%       33.9%         -13.4%      20.7%

No band shows an edge, and the largest disagreements -- the ones that make
the best posts -- have the worst estimate. At a real book the ~4.5% overround
comes off every row. The honest reading is that there is no evidence the
model's disagreements with the market are exploitable; it is a transparent
forecaster that the market beats (2025: market Brier .2109, Elo .2205), not a
tool for beating the book.

The quarterback shift still earns its place on this test. On QB-out games,
backing raw Elo's disagreements returned -20.3% at fair odds, because raw Elo
keeps backing the team that has just lost its quarterback; with the shift it
is -4.0%, inside the noise on 104 bets. It removes a real source of bad picks
even though it does not create good ones.

    python -m pipeline.nfl.backtest_market_edge
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize_scalar

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.backtest_injury_effect import load_games, injury_summary, attach  # noqa: E402
from pipeline.nfl.elo_model import run_elo  # noqa: E402
from pipeline.nfl.games import moneyline_to_prob  # noqa: E402

EPS = 1e-6


def _lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sg(z):
    return 1 / (1 + np.exp(-z))


def build():
    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        P = json.load(f)["elo_params"]
    g = load_games()
    g["elo_pred"] = run_elo(g, k=P["k"], home_adv=P["home_adv"], scale=P["scale"],
                            rest_adv=P.get("rest_adv", 0.0),
                            season_regression=P.get("season_regression", 0.75))
    g = attach(g, injury_summary())
    sc = pl.read_parquet(ROOT / "data" / "nfl" / "schedules.parquet").to_pandas()[
        ["season", "week", "home_team", "away_team", "home_moneyline", "away_moneyline"]]
    g = g.merge(sc, on=["season", "week", "home_team", "away_team"], how="left")
    rh = moneyline_to_prob(g["home_moneyline"].values)
    ra = moneyline_to_prob(g["away_moneyline"].values)
    g["mkt"] = rh / (rh + ra)

    folds = []
    for s in range(2021, int(g["season"].max()) + 1):
        tr, te = g[g["season"] < s], g[g["season"] == s].copy()
        if te.empty:
            continue
        x, d, y = _lg(tr["elo_pred"].values), tr["qb_out_diff"].values, tr["home_win"].values
        b = minimize_scalar(
            lambda b: -np.mean(y * np.log(_sg(x + b * d) + 1e-12) + (1 - y) * np.log(1 - _sg(x + b * d) + 1e-12)),
            bounds=(-3, 3), method="bounded").x
        te["adj"] = _sg(_lg(te["elo_pred"].values) + b * te["qb_out_diff"].values)
        folds.append(te)
    t = pd.concat(folds)
    return t[t["mkt"].notna()]


def lean_returns(t, col, lo, hi=1.0):
    gap = t[col] - t["mkt"]
    sel = (gap.abs() >= lo) & (gap.abs() < hi)
    s, gp = t[sel], gap[sel]
    home = gp > 0
    q = np.where(home, s["mkt"], 1 - s["mkt"])
    won = np.where(home, s["home_win"] == 1, s["home_win"] == 0).astype(float)
    r = won / q - 1
    return len(s), won.mean(), q.mean(), r.mean(), r.std() / np.sqrt(max(len(r), 1))


def main():
    t = build()
    print(f"walk-forward 2021-{int(t['season'].max())}, {len(t)} games with a market price\n")
    print("Adjusted model, betting its side at fair (no-vig) odds:")
    print(f"{'disagreement':>14}{'bets':>7}{'won':>8}{'mkt expected':>14}{'fair ROI':>10}{'+/- 1 s.e.':>12}")
    for lo, hi in [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]:
        n, w, q, r, se = lean_returns(t, "adj", lo, hi)
        lab = f"{int(lo*100)}-{int(hi*100)} pts" if hi < 1 else f"{int(lo*100)}+ pts"
        print(f"{lab:>14}{n:>7}{w:>8.1%}{q:>14.1%}{r:>+10.1%}{se:>11.1%}")

    aff = t["qb_out_diff"] != 0
    print("\nQB-out games only, 5+ pt disagreements:")
    for col, lab in [("elo_pred", "raw Elo"), ("adj", "with QB shift")]:
        n, w, q, r, se = lean_returns(t[aff], col, 0.05)
        print(f"  {lab:16s} bets {n:4d}  won {w:.1%} vs {q:.1%} expected  fair ROI {r:+.1%} (+/- {se:.1%})")
    print("\nNo vig included. A real book takes roughly 4.5% off every figure above.")


if __name__ == "__main__":
    main()

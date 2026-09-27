"""Does recalibrating the model's probabilities make them better, out of sample?

WHY. The public track record showed the model overconfident through the
middle of its range (said 53%, happened 43.5%, across 4,534 predictions), and
the same-game parlay deliberately picks legs at 64-71%, which is exactly the
range in question. Before changing any probability on the site, this asks the
only question that matters: if a correction had been fitted on the days we
had at the time, would the NEXT day's probabilities have been better?

WHAT IS CALIBRATED, and what deliberately is not.

  MLB props at their posted line. These carry real spread (Hits runs from
  0.19 to 0.51 between the 10th and 90th percentile), so there is something
  for a calibration map to act on.

  NFL ladder RUNGS, not NFL base lines. An NFL yardage line is set at the
  model's own projection, so "P(over my own mean)" is a near-constant fixed by
  the shape of the residual distribution: Receiving Yds is 0.375 +/- 0.003
  across 360 props, Passing Yds 0.500 +/- 0.001. There is nothing there to
  recalibrate beyond moving a constant, which can take skill to zero and no
  higher. The rungs are where the model actually varies, and they are what the
  same-game parlay and the suggested parlay put on a card.

  Only snapshots from the CURRENT probability model. Finalized snapshots are
  frozen, and the first ten MLB days (Jul 20 - Aug 2) came from an earlier
  over/under model. Calibrating on them would fit a correction to a model that
  no longer exists.

METHOD. Platt scaling per market: a logistic regression of the outcome on
logit(p), two parameters, which is the standard choice when some markets have
only a few hundred graded predictions. A slope below 1 pulls probabilities in
toward the middle, which is exactly the "too spread" pattern the track record
shows. Isotonic regression is reported alongside as a check, not as a
contender to be picked per market, because choosing the winner per market on
the test data would itself be fitting to the test.

PROTOCOL. Strictly walk-forward. For every test day, the calibrator is fitted
only on days before it, then applied to that day. Nothing is ever scored on
data its own correction saw.

    python -m pipeline.common.backtest_calibration
"""
import collections
import glob
import json
import pathlib
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"

MLB_CURRENT_MODEL = "negbin-v1"
MIN_TRAIN = 150          # graded predictions before a market gets a correction
EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_mlb():
    """(market, date, prob, outcome) for every graded MLB prop from the
    current probability model, plus moneylines."""
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "mlb_2026-*.json"))):
        s = json.load(open(f, encoding="utf-8"))
        d = s.get("date")
        for g in s.get("games") or []:
            if (g.get("already_played") and g.get("home_score") is not None
                    and g.get("away_score") is not None and g.get("elo_home_prob") is not None):
                rows.append(("MLB Moneyline", d, g["elo_home_prob"],
                             1 if g["home_score"] > g["away_score"] else 0))
            if s.get("prop_prob_model") != MLB_CURRENT_MODEL:
                continue
            for p in g.get("props") or []:
                if p.get("actual") is None:
                    continue
                hr = p["market"] == "Anytime HR"
                pr = p.get("model_prob") if hr else p.get("model_over_prob")
                if pr is None or (not hr and (p.get("line") is None or p["line"] <= 0)):
                    continue
                out = (p["actual"] > 0) if hr else (p["actual"] > p["line"])
                rows.append(("MLB " + p["market"], d, pr, 1 if out else 0))
    return rows


def load_nfl():
    """(market, week-key, prob, outcome) for every graded NFL ladder rung and
    Anytime TD. Keyed by (season, week) so walk-forward steps a week at a time."""
    rows = []
    for f in glob.glob(str(RESULTS / "nfl_*.json")):
        s = json.load(open(f, encoding="utf-8"))
        if not s.get("graded"):
            continue
        name = pathlib.Path(f).stem          # nfl_2026_wk01_SF_LA
        parts = name.split("_")
        key = f"{parts[1]}-{parts[2]}"       # 2026-wk01
        for p in s.get("props_graded") or []:
            if p.get("market") == "Anytime TD":
                if p.get("hit") is not None and p.get("model_prob") is not None:
                    rows.append(("NFL Anytime TD", key, p["model_prob"], 1 if p["hit"] else 0))
                continue
            if p.get("actual_value") is None or not p.get("ladder"):
                continue
            for r in p["ladder"]:
                rows.append(("NFL " + p["market"] + " (rungs)", key, r["over_prob"],
                             1 if p["actual_value"] > r["line"] else 0))
    return rows


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def skill(p, y, base):
    """Brier skill against quoting a fixed base rate. The base rate is passed
    in so raw and calibrated are scored against the SAME yardstick."""
    ns = base * (1 - base)
    return 1 - brier(p, y) / ns if ns > 0 else float("nan")


def fit_platt(p, y):
    m = LogisticRegression(C=1e6)
    m.fit(_logit(p).reshape(-1, 1), y)
    return m


def apply_platt(m, p):
    return m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def walk_forward(rows):
    """Per market: step through periods in order, fit on everything before,
    score the period. Returns {market: dict of pooled out-of-sample results}."""
    by_mk = collections.defaultdict(list)
    for mk, period, p, y in rows:
        by_mk[mk].append((period, p, y))

    out = {}
    for mk, items in sorted(by_mk.items()):
        periods = sorted(set(i[0] for i in items))
        raw, platt, iso, ys = [], [], [], []
        slopes = []
        for per in periods:
            train = [(p, y) for (q, p, y) in items if q < per]
            test = [(p, y) for (q, p, y) in items if q == per]
            if len(train) < MIN_TRAIN or not test:
                continue
            tp, ty = np.array([t[0] for t in train]), np.array([t[1] for t in train])
            if ty.min() == ty.max():
                continue
            xp, xy = np.array([t[0] for t in test]), np.array([t[1] for t in test])
            m = fit_platt(tp, ty)
            slopes.append(float(m.coef_[0][0]))
            ir = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(tp, ty)
            raw.append(xp); platt.append(apply_platt(m, xp)); iso.append(ir.predict(xp)); ys.append(xy)
        if not ys:
            continue
        raw, platt, iso, ys = map(np.concatenate, (raw, platt, iso, ys))
        base = float(ys.mean())
        out[mk] = {
            "n": len(ys), "periods": len(slopes),
            "raw_skill": skill(raw, ys, base),
            "platt_skill": skill(platt, ys, base),
            "iso_skill": skill(iso, ys, base),
            "raw_brier": brier(raw, ys), "platt_brier": brier(platt, ys),
            "slope": float(np.median(slopes)),
            "raw": raw, "platt": platt, "y": ys,
        }
    return out


def calibration_rows(p, y, lo=0.6, hi=0.72):
    """Said vs happened inside one band -- the SGP band by default."""
    m = (p >= lo) & (p < hi)
    if m.sum() < 20:
        return None
    return int(m.sum()), float(p[m].mean()), float(y[m].mean())


def main():
    results = {}
    results.update(walk_forward(load_mlb()))
    results.update(walk_forward(load_nfl()))

    print("Walk-forward, fitted only on earlier days/weeks, scored out of sample.")
    print("Skill = Brier skill vs quoting the test set's own base rate.\n")
    print(f"{'market':34s}{'n':>7}{'raw':>9}{'platt':>9}{'isotonic':>10}{'gain':>8}{'slope':>7}")
    for mk, r in results.items():
        gain = r["platt_skill"] - r["raw_skill"]
        print(f"{mk:34s}{r['n']:>7}{r['raw_skill']*100:>8.1f}%{r['platt_skill']*100:>8.1f}%"
              f"{r['iso_skill']*100:>9.1f}%{gain*100:>+7.1f}{r['slope']:>7.2f}")

    print("\nThe SGP band, 60-72% as shown, out of sample:")
    for mk, r in results.items():
        if "rungs" not in mk:
            continue
        a = calibration_rows(r["raw"], r["y"])
        b = calibration_rows(r["platt"], r["y"])
        if a:
            print(f"  {mk:30s} raw: said {a[1]:.1%} happened {a[2]:.1%} (n={a[0]})"
                  + (f" | calibrated: said {b[1]:.1%} happened {b[2]:.1%} (n={b[0]})" if b else ""))
    return results


if __name__ == "__main__":
    main()

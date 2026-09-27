"""Recalibrate NFL yardage-prop probabilities against how often they actually hit.

WHAT WAS WRONG. Graded NFL ladder rungs ran hot at the top of the range:
rungs the model called 66% came in 58%, 75% came in 66%, 84% came in 74%.
That is exactly where the same-game parlay shops, since its legs have to
land at 64-71%, so a card built from them promised more than it delivered.

WHAT THIS DOES. Fits one Platt correction (a logistic regression of the
outcome on logit(p), two parameters) over every graded yardage rung, and
applies it to both the rungs and the posted-line probability of each yardage
prop. One pooled correction rather than one per market: only a few weeks are
graded, Passing Yds and Rush + Rec Yds have too few rungs to fit alone, and
the four markets bend the same way (fitted slopes 0.59-0.75 where there was
enough data to fit them separately).

EVIDENCE, from pipeline/common/backtest_calibration.py, walk-forward (fitted
only on earlier weeks, scored on the next):

    skill on rungs           +7.6%  ->  +8.5%
    SGP band, as shown       said 66.0%, happened 58.2%  (n=306)
    SGP band, calibrated     said 65.4%, happened 65.0%  (n=334)

WHAT IT DELIBERATELY DOES NOT TOUCH.
  Anytime TD. It is the one NFL market with proven skill, and the same
  backtest made it worse (+4.6% -> -0.3%). Left raw.
  The count markets (Carries, Receptions, Completions, Pass Attempts, Passing
  TDs, kicking). No ladder, and no validated correction.
  MLB entirely. Finalized MLB snapshots are regenerated three days after the
  game from game logs that already include it, so every stored MLB prop
  probability leans toward its own result (correlation +0.51 between how far
  the trailing average moved and what the batter then did). A correction
  fitted on that would be fitted to the leak.

FITTED ON RAW, ALWAYS. Every value this changes keeps its original alongside
(model_over_prob_raw, ladder[].over_prob_raw), and the fitter reads the raw
field when present. Otherwise the next run would fit a correction to the
previous run's corrected output and compound it.
"""
import json
import pathlib

import numpy as np

YARDAGE_MARKETS = {"Passing Yds", "Rushing Yds", "Receiving Yds", "Rush + Rec Yds"}
VERSION_SUFFIX = "-cal1"
MIN_FIT = 500      # graded rungs before any correction is trusted
EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_graded_rungs(results_dir):
    """(raw probability, outcome) for every graded NFL yardage ladder rung.

    Snapshots are frozen pregame (write_prediction_snapshot never rewrites a
    game once it has one, and grading runs before generation), so these are
    genuinely out-of-sample predictions, unlike the MLB snapshots."""
    rows = []
    for f in pathlib.Path(results_dir).glob("nfl_*.json"):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not s.get("graded"):
            continue
        for p in s.get("props_graded") or []:
            if p.get("market") not in YARDAGE_MARKETS:
                continue
            actual = p.get("actual_value")
            if actual is None or not p.get("ladder"):
                continue
            for r in p["ladder"]:
                prob = r.get("over_prob_raw", r.get("over_prob"))
                if prob is None:
                    continue
                rows.append((float(prob), 1 if actual > r["line"] else 0))
    return rows


class RungCalibrator:
    def __init__(self, slope, intercept, n):
        self.slope, self.intercept, self.n = float(slope), float(intercept), int(n)

    def __call__(self, p):
        z = self.slope * _logit(p) + self.intercept
        return float(1.0 / (1.0 + np.exp(-z)))

    def describe(self):
        return (f"slope {self.slope:.3f}, intercept {self.intercept:+.3f}, fitted on "
                f"{self.n:,} graded rungs")


def fit(results_dir):
    """The pooled correction, or None if there is not yet enough graded data
    to trust one -- in which case nothing is changed and nothing is stamped."""
    from sklearn.linear_model import LogisticRegression

    rows = load_graded_rungs(results_dir)
    if len(rows) < MIN_FIT:
        print(f"  [calibration] {len(rows)} graded rungs, need {MIN_FIT}: leaving NFL probabilities raw")
        return None
    p = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    if y.min() == y.max():
        return None
    m = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
    cal = RungCalibrator(m.coef_[0][0], m.intercept_[0], len(rows))
    print(f"  [calibration] NFL yardage rungs: {cal.describe()}")
    return cal


def apply(props, cal):
    """Calibrate yardage props in place, keeping every raw value alongside.

    Idempotent: a prop that already carries model_over_prob_raw is skipped,
    so a second pass can never calibrate an already-calibrated number."""
    if cal is None:
        return 0
    n = 0
    for p in props or []:
        if p.get("market") not in YARDAGE_MARKETS or "model_over_prob_raw" in p:
            continue
        if p.get("model_over_prob") is not None:
            p["model_over_prob_raw"] = p["model_over_prob"]
            p["model_over_prob"] = round(cal(p["model_over_prob"]), 3)
        for r in p.get("ladder") or []:
            if r.get("over_prob") is None:
                continue
            r["over_prob_raw"] = r["over_prob"]
            r["over_prob"] = round(cal(r["over_prob"]), 3)
        n += 1
    return n

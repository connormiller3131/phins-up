"""Adjust NFL win probability for a starting quarterback ruled Out.

WHY. The betting market beats raw Elo on NFL games (2025 holdout, 284 games:
market Brier .2109, Elo .2205), and the single largest thing the market knows
that Elo cannot is who is actually playing quarterback. Elo only sees past
results, so it prices a team at full strength the week its starter sits.

EVIDENCE. Tested in pipeline/nfl/backtest_injury_effect.py (August 2026) and
never shipped until now. Re-run on current data, walk-forward pooled over
2021-2026 (train on all earlier seasons, test on the next), 1,392 games, 177
with a QB-out imbalance:

                              all games   QB-out games   other games
    raw Elo                     .2247        .2119          .2265
    full refit (as validated)   .2237        .2008          .2270
    QB shift only (shipped)     .2232        .1999          .2265

WHY THE SHIFT AND NOT THE MODEL THAT WAS VALIDATED. The August model was a
logistic regression on [elo_pred, qb_out_diff], which also re-weights Elo
itself, so it changes every game rather than only the ~13% with a quarterback
out -- and on those other games it is slightly worse (.2270 vs .2265). The
shift adds only the quarterback term in logit space, so any game without a
QB-out imbalance comes back as exactly raw Elo. It wins on every column.

The fitted shift is stable across folds, -0.49 to -0.57 logit per QB out.
In plain terms: a 60% home favourite whose quarterback is out drops to 48%.

THE FEATURE IS COMPUTED EXACTLY AS VALIDATED. injury_summary() and the
report-status rule come straight from the backtest module, so "ruled Out" and
"QB" mean the same thing live as they did in the test: report_status "Out",
position "QB", counted per team-week, home minus away. Doubtful and
Questionable are ignored, as they were in the test; skill-position absences
are ignored because they added nothing (.2008 -> .2015).

ONE DELIBERATE EXTENSION. A manual QB entry in starter_overrides.json also
counts as the team's starter being out, because an override exists precisely
to say the depth chart's quarterback is not starting. That is the fact this
feature models, but it was not in the backtest -- and a benching can move a
team less than an injury, since a benched starter was often playing badly. It
is capped with max(), so a quarterback both ruled Out and overridden is never
counted twice.

TIMING. Out designations mostly arrive late in the week, while a game's graded
snapshot freezes the first time its week is current, usually a Tuesday. So the
game-day number users see (and the parlays built from it) gets the full
adjustment, while the frozen record often predates the designation and
under-credits it. The freeze is left alone: it is what keeps the record free
of hindsight.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EPS = 1e-6


def _logit(p):
    p = min(max(float(p), EPS), 1 - EPS)
    return np.log(p / (1 - p))


def _sig(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def fit_shift():
    """The logit shift per quarterback out, fitted on every completed game in
    the injury era. Returns (beta, n_games, n_affected), or (0.0, ...) if the
    data cannot support a fit, in which case nothing is adjusted."""
    from scipy.optimize import minimize_scalar
    from pipeline.nfl.backtest_injury_effect import load_games, injury_summary, attach
    from pipeline.nfl.elo_model import run_elo

    with open(ROOT / "notebooks_out" / "nfl_win_prob_backtest.json") as f:
        P = json.load(f)["elo_params"]
    g = load_games()
    g["elo_pred"] = run_elo(g, k=P["k"], home_adv=P["home_adv"], scale=P["scale"],
                            rest_adv=P.get("rest_adv", 0.0),
                            season_regression=P.get("season_regression", 0.75))
    g = attach(g, injury_summary())

    x = np.log(np.clip(g["elo_pred"].values, EPS, 1 - EPS) / (1 - np.clip(g["elo_pred"].values, EPS, 1 - EPS)))
    d = g["qb_out_diff"].values
    y = g["home_win"].values
    n_aff = int((d != 0).sum())
    if n_aff < 50:
        print(f"  [qb adjust] only {n_aff} QB-out games to fit on: leaving win probabilities raw")
        return 0.0, len(g), n_aff

    def loss(b):
        p = 1.0 / (1.0 + np.exp(-(x + b * d)))
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

    beta = float(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x)
    print(f"  [qb adjust] shift {beta:+.3f} logit per QB out, fitted on {len(g):,} games "
          f"({n_aff} with a QB-out imbalance)")
    return beta, len(g), n_aff


def qb_out_by_week(season, overrides_by_week=None):
    """{(week, team): starting quarterbacks out} for one season.

    From the injury report exactly as validated, then raised to 1 for any team
    whose quarterback was manually overridden that week (see the module note
    on why that is an extension, and why it is capped rather than added)."""
    from pipeline.nfl.backtest_injury_effect import injury_summary

    inj = injury_summary()
    inj = inj[inj["season"] == season]
    out = {(int(r.week), r.team): float(r.qb_out) for r in inj.itertuples(index=False)
           if r.qb_out}
    for week, teams in (overrides_by_week or {}).items():
        for team, positions in teams.items():
            if "QB" in positions:
                out[(int(week), team)] = max(out.get((int(week), team), 0.0), 1.0)
    return out


def adjust(elo_home_prob, qb_out_diff, beta):
    """Elo home win probability with the quarterback shift applied. A game with
    no imbalance returns the input unchanged, to the last digit."""
    if elo_home_prob is None or not qb_out_diff or not beta:
        return elo_home_prob
    return _sig(_logit(elo_home_prob) + beta * qb_out_diff)

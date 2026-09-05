"""Does the anytime-TD model need an opportunity feature?

Every other NFL prop was tested for this in backtest_count_volume.py and
most kept it; anytime_td was never in that script's STATS dict, so the
question was never asked for the one market the site promotes hardest (the
TD SPECIAL card). The deployed model sees a player's trailing TD rate, the
defense's trailing allowed rate, weather, rest and the implied team total --
but nothing about how much work he actually gets. A TE2 who scored in 4 of
his last 8 games on ~2 touches a game and an every-down receiver on ~9
touches both present as "0.500", and the model has no way to tell them
apart.

Tested walk-forward on the untouched 2025 season (train strictly earlier,
refit per week -- same convention as backtest_props.py /
backtest_count_volume.py). Feature sets:

  baseline      the deployed seven (prop_models.FEATURES)
  +opportunity  the same seven plus own_trailing_volume, built from
                prop_data's "opportunities" column (carries + targets)

Three scorings, because Brier alone would not settle the question that
actually prompted this:

1. Brier / log loss over every row -- the standard check, but TD is a
   ~22% base-rate market, so a whole-population Brier moves very little
   even when the tail is badly wrong.
2. Calibration by usage tier -- predicted vs realized TD rate for low,
   mid and high trailing-opportunity players. This is the direct test of
   the complaint: if the baseline model's low-usage bucket predicts well
   above what those players actually do, it is systematically
   manufacturing edge on depth pieces.
3. A simulated TD SPECIAL -- each test week, take the model's top 3 picks
   on distinct teams (exactly collectNflTdSpecialLegs' model-only path)
   and score the realized TD rate of those picks. This is the product
   metric: it answers "would the card have been right more often", which
   is not the same question as "is the model better on average".
"""
import sys
import re
import pathlib
import json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.props.prop_data import build_prop_table
from pipeline.nfl.props.prop_models import FEATURES, walk_forward_anytime_td
from pipeline.common.metrics import brier_score, log_loss, calibration_curve

TEST_SEASONS = [2025]
POSITIONS = ["RB", "WR", "TE"]
SPECIAL_LEGS = 3          # matches collectNflTdSpecialLegs' 3-leg card
USAGE_TIERS = [(0, 4), (4, 8), (8, 99)]   # trailing opportunities per game


def _tier_label(lo, hi):
    return f"{lo}-{hi} opp/gm" if hi < 99 else f"{lo}+ opp/gm"


def evaluate(df, pred_col):
    scored = df[df[pred_col].notna()]
    y, p = scored["actual"].values, scored[pred_col].values
    return {
        "n": int(len(scored)),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "base_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "calibration": calibration_curve(y, p, n_bins=10),
    }


def usage_tiers(df, pred_col):
    """Predicted vs realized TD rate, split by how much work the player
    actually gets. own_trailing_volume is present on the frame regardless of
    which feature set produced pred_col, so both models are sliced the same
    way and the tiers stay comparable."""
    scored = df[df[pred_col].notna()]
    out = []
    for lo, hi in USAGE_TIERS:
        m = (scored["own_trailing_volume"] >= lo) & (scored["own_trailing_volume"] < hi)
        if m.sum() == 0:
            continue
        y, p = scored.loc[m, "actual"].values, scored.loc[m, pred_col].values
        out.append({
            "tier": _tier_label(lo, hi),
            "n": int(m.sum()),
            "predicted_mean": float(p.mean()),
            "actual_mean": float(y.mean()),
            "overstatement": float(p.mean() - y.mean()),
            "brier": brier_score(y, p),
        })
    return out


def simulated_special(df, pred_col):
    """Replays the TD SPECIAL's own selection rule week by week: highest
    model probability first, at most one leg per team, 3 legs. Reports how
    often those picks actually scored, plus who they were."""
    scored = df[df[pred_col].notna()]
    hits = total = 0
    picks_by_week = []
    for (season, week), grp in scored.groupby(["season", "week"]):
        used, legs = set(), []
        for r in grp.sort_values(pred_col, ascending=False).itertuples(index=False):
            if len(legs) >= SPECIAL_LEGS:
                break
            if r.team in used:
                continue
            used.add(r.team)
            legs.append(r)
        if not legs:
            continue
        hits += sum(int(r.actual) for r in legs)
        total += len(legs)
        picks_by_week.append({
            "season": int(season), "week": int(week),
            "legs": [{"player": r.player_display_name, "team": r.team,
                      "pred": round(float(getattr(r, pred_col)), 3),
                      "opp_per_gm": round(float(r.own_trailing_volume), 1),
                      "scored": int(r.actual)} for r in legs],
        })
    return {
        "legs_graded": total,
        "legs_hit": hits,
        "hit_rate": (hits / total) if total else None,
        "all_three_weeks": sum(1 for w in picks_by_week if all(leg["scored"] for leg in w["legs"])),
        "weeks": picks_by_week,
    }


def main():
    # Built with the volume column so own_trailing_volume exists for BOTH
    # feature sets -- the baseline simply doesn't train on it. Same rows
    # either way (volume columns are never NaN for a logged game), so the
    # two models are scored on an identical population rather than on
    # populations that differ by which rows survived a dropna.
    df = build_prop_table("anytime_td", POSITIONS, volume_col="opportunities")
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    print(f"TD rows: {len(df)}  seasons {df['season'].min()}-{df['season'].max()}")
    print(f"test seasons: {TEST_SEASONS}\n")

    feature_sets = {
        "baseline": FEATURES,
        "+opportunity": FEATURES + ["own_trailing_volume"],
    }

    results = {}
    for name, feats in feature_sets.items():
        print(f"--- {name} ({len(feats)} features) ---", flush=True)
        # Sanitised because simulated_special reads predictions off
        # DataFrame.itertuples, which cannot expose an attribute whose name
        # is not a valid Python identifier ("pred_+opportunity" is not).
        col = "pred_" + re.sub(r"\W", "_", name)
        df[col] = walk_forward_anytime_td(df, TEST_SEASONS, features=feats)
        overall = evaluate(df, col)
        tiers = usage_tiers(df, col)
        special = simulated_special(df, col)
        results[name] = {"features": feats, "overall": overall, "usage_tiers": tiers,
                         "simulated_special": special}
        print(f"  n={overall['n']}  brier {overall['brier']:.4f}  log loss {overall['log_loss']:.4f}"
              f"  (base rate {overall['base_rate']:.3f}, mean pred {overall['mean_pred']:.3f})")
        for t in tiers:
            print(f"    {t['tier']:<14} n={t['n']:<6} pred {t['predicted_mean']:.3f}"
                  f"  actual {t['actual_mean']:.3f}  overstates by {t['overstatement']:+.3f}")
        print(f"  simulated TD SPECIAL: {special['legs_hit']}/{special['legs_graded']} legs hit"
              f" ({special['hit_rate']:.3f}), all-3 weeks: {special['all_three_weeks']}")
        print()

    b, v = results["baseline"], results["+opportunity"]
    print("=== summary ===")
    print(f"  brier            {b['overall']['brier']:.4f} -> {v['overall']['brier']:.4f}")
    print(f"  log loss         {b['overall']['log_loss']:.4f} -> {v['overall']['log_loss']:.4f}")
    lo_b = next((t for t in b["usage_tiers"] if t["tier"].startswith("0-")), None)
    lo_v = next((t for t in v["usage_tiers"] if t["tier"].startswith("0-")), None)
    if lo_b and lo_v:
        print(f"  low-usage overstatement  {lo_b['overstatement']:+.3f} -> {lo_v['overstatement']:+.3f}")
    print(f"  TD SPECIAL hit rate      {b['simulated_special']['hit_rate']:.3f} ->"
          f" {v['simulated_special']['hit_rate']:.3f}")

    path = ROOT / "notebooks_out" / "nfl_td_volume_backtest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

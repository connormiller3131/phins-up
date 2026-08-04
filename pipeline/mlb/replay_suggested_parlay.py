"""Reconstructs and grades the MLB Suggested Parlay for every finalized day.

There is no stored parlay history: the card is built client-side at render
time (dashboard_live.html's buildRealisticParlay), so nothing was ever
written to disk. It is recoverable anyway, because two things are true:

1. The selection is deterministic. buildRealisticParlay / bestPropInMarket /
   pickLine are pure sorts over the day's props with no randomness anywhere.
   (The Flin Flon Special IS random by design and is deliberately excluded
   here -- it re-draws on every page load and was never a stable daily pick.)
2. Every finalized day snapshot in docs/results/ holds the exact props array
   the card was built from, with real `actual` values already attached by
   attach_prop_actuals.

So replaying the same logic over a frozen snapshot rebuilds that day's real
card and grades it.

One honest caveat on fidelity: a snapshot is the FINALIZED version of the
day. Confirmed-lineup flags land during the afternoon as real lineups post,
so a card reconstructed here matches what the site showed once lineups were
in -- not necessarily what a visitor saw at 9am, before any lineup existed.
Legs are graded only where a real `actual` exists; a player who never
appeared is skipped rather than counted as a loss, matching grade_props.py's
own "did not play isn't a fair test" rule.
"""
import sys
import json
import pathlib
import argparse

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "docs" / "results"
RELIABLE_TRAILING_N = 8            # mirrors dashboard_live.html
BINARY_MARKETS = {"Anytime HR", "Anytime TD"}
FILLER_MARKETS = ["Anytime HR", "Hits", "RBI", "Walks"]


def implied_prob(odds):
    return (-odds) / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def no_vig_home(ml_home, ml_away):
    ph, pa = implied_prob(ml_home), implied_prob(ml_away)
    return ph / (ph + pa)


def pick_line(p):
    """Port of pickLine: only ever bumps UP from the base line, to the
    HIGHEST ladder rung that still clears 50%."""
    if p.get("dk_implied_prob") is not None or not p.get("ladder"):
        return p.get("line"), p.get("model_over_prob")
    higher = sorted([r for r in p["ladder"] if r["line"] > p["line"]],
                    key=lambda r: -r["line"])
    for r in higher:
        if r["over_prob"] >= 0.5:
            return r["line"], r["over_prob"]
    return p.get("line"), p.get("model_over_prob")


def is_reliable(leg):
    return leg["trailing_n"] is None or leg["trailing_n"] >= RELIABLE_TRAILING_N


def best_prop_in_market(props, market, used):
    """Port of bestPropInMarket, including its exact tiebreak order:
    real-edge beats model-only; among model-only, reliable beats unreliable;
    then rank by edge / projected production / probability."""
    is_binary = market in BINARY_MARKETS
    best = None
    for p in props:
        if p.get("market") != market or p.get("player_id") in used:
            continue
        if p.get("__lineup_known") and p.get("confirmed_starter") is False:
            continue
        model_p = p.get("model_prob") if is_binary else p.get("model_over_prob")
        if model_p is None:
            continue
        if not is_binary and (p.get("line") is None or p["line"] <= 0):
            continue

        if is_binary:
            chosen_line, chosen_prob = None, model_p
        else:
            chosen_line, chosen_prob = pick_line(p)

        dk = p.get("dk_implied_prob")
        leg = {
            "edge": (model_p - dk) if dk is not None else None,
            "prob": model_p if dk is not None else chosen_prob,
            "label": p.get("player"), "market": market,
            "player_id": p.get("player_id"), "projected": p.get("projected"),
            "trailing_n": p.get("trailing_n"), "line": chosen_line,
            "actual": p.get("actual"), "is_binary": is_binary,
        }
        if leg["edge"] is not None and leg["edge"] <= 0:
            continue
        if leg["edge"] is None and not is_binary and (leg["prob"] is None or leg["prob"] < 0.5):
            continue

        if best is None:
            best = leg
            continue
        leg_real, best_real = leg["edge"] is not None, best["edge"] is not None
        if leg_real != best_real:
            if leg_real:
                best = leg
            continue
        if not leg_real:
            lr, br = is_reliable(leg), is_reliable(best)
            if lr != br:
                if lr:
                    best = leg
                continue
        if leg_real:
            lrank, brank = leg["edge"], best["edge"]
        elif not is_binary:
            lrank, brank = leg["projected"], best["projected"]
        else:
            lrank, brank = leg["prob"], best["prob"]
        if lrank is not None and brank is not None and lrank > brank:
            best = leg
    return best


def build_parlay(games, max_moneylines=2):
    """Port of buildRealisticParlay(games, {maxMoneylines: 2})."""
    legs, used = [], set()

    ml_candidates = []
    for g in games:
        m = g.get("market")
        if not m or m.get("mlHome") is None or m.get("mlAway") is None:
            continue
        fair_home = no_vig_home(m["mlHome"], m["mlAway"])
        home_edge = g["elo_home_prob"] - fair_home
        base = {"market": "Moneyline", "player_id": None, "trailing_n": None,
                "home_score": g.get("home_score"), "away_score": g.get("away_score"),
                "is_binary": False, "line": None}
        if home_edge > 0:
            ml_candidates.append({**base, "edge": home_edge, "prob": g["elo_home_prob"],
                                  "label": f"{g['homeAbbr']} ML", "side": "home"})
        elif -home_edge > 0:
            ml_candidates.append({**base, "edge": -home_edge, "prob": 1 - g["elo_home_prob"],
                                  "label": f"{g['awayAbbr']} ML", "side": "away"})
        else:
            home_fav = g["elo_home_prob"] >= 0.5
            ml_candidates.append({**base, "edge": None,
                                  "prob": g["elo_home_prob"] if home_fav else 1 - g["elo_home_prob"],
                                  "label": f"{g['homeAbbr'] if home_fav else g['awayAbbr']} ML",
                                  "side": "home" if home_fav else "away"})

    ml_candidates.sort(key=lambda c: (0 if c["edge"] is not None else 1,
                                      -(c["edge"] if c["edge"] is not None else c["prob"])))
    legs.extend(ml_candidates[:max_moneylines])

    all_props = []
    for g in games:
        props = g.get("props") or []
        lineup_known = any(p.get("section") == "Batting" and p.get("confirmed_starter") is True
                           for p in props)
        for p in props:
            q = dict(p)
            if p.get("section") == "Batting":
                q["__lineup_known"] = lineup_known
            all_props.append(q)

    tb = best_prop_in_market(all_props, "Total Bases", used)
    if tb:
        legs.append(tb)
        used.add(tb["player_id"])

    if len(legs) < 4:
        fillers = [best_prop_in_market(all_props, mk, used) for mk in FILLER_MARKETS]
        fillers = [f for f in fillers if f]
        fillers.sort(key=lambda f: (0 if f["edge"] is not None else 1,
                                    -(f["edge"] if f["edge"] is not None else f["prob"])))
        for f in fillers:
            if len(legs) >= 4:
                break
            legs.append(f)
            used.add(f["player_id"])
    return legs


def grade_leg(leg):
    """True/False, or None when the leg cannot be fairly graded."""
    if leg["market"] == "Moneyline":
        hs, as_ = leg.get("home_score"), leg.get("away_score")
        if hs is None or as_ is None or hs == as_:
            return None
        home_won = hs > as_
        return home_won if leg["side"] == "home" else (not home_won)
    actual = leg.get("actual")
    if actual is None:
        return None            # did not appear -- not a fair test
    if leg["is_binary"]:
        return actual > 0
    if leg.get("line") is None:
        return None
    return actual > leg["line"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print every leg")
    args = ap.parse_args()

    index = json.loads((RESULTS_DIR / "mlb_index.json").read_text(encoding="utf-8"))
    dates = index.get("dates", [])

    day_rows, leg_total, leg_hit, by_market = [], 0, 0, {}
    parlays_all_hit = 0
    parlays_graded = 0

    for d in dates:
        path = RESULTS_DIR / f"mlb_{d}.json"
        if not path.exists():
            continue
        day = json.loads(path.read_text(encoding="utf-8"))
        legs = build_parlay(day.get("games", []))
        if not legs:
            continue

        graded = [(leg, grade_leg(leg)) for leg in legs]
        scored = [(leg, r) for leg, r in graded if r is not None]
        hits = sum(1 for _, r in scored if r)
        if scored:
            parlays_graded += 1
            if hits == len(scored):
                parlays_all_hit += 1
        leg_total += len(scored)
        leg_hit += hits
        for leg, r in scored:
            b = by_market.setdefault(leg["market"], {"n": 0, "hit": 0})
            b["n"] += 1
            b["hit"] += int(r)

        day_rows.append((d, len(legs), len(scored), hits))
        if args.verbose:
            print(f"\n{d}")
            for leg, r in graded:
                mark = "-" if r is None else ("HIT " if r else "miss")
                line = f" {leg['line']}+" if leg.get("line") is not None else ""
                src = f"edge {leg['edge']:+.3f}" if leg["edge"] is not None else f"model {leg['prob']:.3f}"
                print(f"   [{mark}] {leg['market']:<14}{leg['label']}{line}  ({src})")

    print("\n=== MLB Suggested Parlay, replayed over every finalized day ===")
    print(f"{'date':<12}{'legs':>5}{'graded':>8}{'hits':>6}")
    for d, n, g, h in day_rows:
        print(f"{d:<12}{n:>5}{g:>8}{h:>6}")

    if leg_total:
        print(f"\nLeg record: {leg_hit}/{leg_total} ({leg_hit/leg_total:.1%})")
        print(f"Full parlays that hit every graded leg: {parlays_all_hit}/{parlays_graded}")
        print("\nBy market:")
        for mk in sorted(by_market):
            b = by_market[mk]
            print(f"  {mk:<16} {b['hit']:>3}/{b['n']:<3} ({b['hit']/b['n']:.1%})")
    else:
        print("\nNo gradeable legs found.")


if __name__ == "__main__":
    main()

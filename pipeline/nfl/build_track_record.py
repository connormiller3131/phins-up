"""Build docs/track-record/index.html -- a public, crawlable page showing how
accurate the model has actually been.

WHY THIS PAGE EXISTS. Almost every product in this category claims a win rate
and shows nothing. This site publishes its own error rate, including where the
betting market beats it. That is the single most persuasive thing it has, and
until now it lived inside a collapsed <details> on the MLB tab -- unreachable
by a link, invisible to search, and impossible to point at when someone says
"prove it".

THE MLB GRADING RULES ARE MIRRORED, NOT INVENTED. collect_mlb computes every
number the same way loadMlbTrackRecord() in dashboard_live.html does, from the
same finalized snapshots. That duplication is a real risk -- two languages,
one definition -- and it was verified by running the site's own JavaScript
against the live deployment and comparing every market to this module's output
(identical across all 11, 38,230 predictions). There is no automated check;
if you change a rule on either side, redo that comparison by hand.

THE NFL SIDE IS SCORED DIFFERENTLY FROM ITS OWN STORED `hit`, deliberately.
grade_results.py grades a laddered prop against the MIDDLE RUNG of its ladder,
while model_over_prob is the probability of clearing the prop's own `line`,
and those are different numbers on 260 of 268 laddered props. Scoring one
against the other would pair a probability for one threshold with an outcome
for another, so collect_nfl recomputes from actual_value against the base
line. Anytime TD has no line and uses `hit` directly.

Baseline matters more than accuracy. "Anytime HR is 89% accurate" sounds
excellent until you notice that always guessing "no" scores 89% too. Every row
carries the naive baseline next to it for exactly that reason.
"""
import json
import pathlib

from pipeline.nfl.build_game_pages import SITE, head, FOOT, e, pct


def _bucket(store, market):
    return store.setdefault(market, {"n": 0, "correct": 0, "actual_sum": 0.0,
                                     "prob_sum": 0.0, "brier_sum": 0.0})


def _record(store, market, prob, outcome, cal=None):
    b = _bucket(store, market)
    b["n"] += 1
    b["correct"] += 1 if (1 if prob >= 0.5 else 0) == outcome else 0
    b["actual_sum"] += outcome
    b["prob_sum"] += prob
    b["brier_sum"] += (prob - outcome) ** 2
    if cal is not None:
        idx = min(9, max(0, int(prob * 10)))
        c = cal[idx]
        c["n"] += 1
        c["pred_sum"] += prob
        c["actual_sum"] += outcome


def new_calibration():
    return [{"lo": i / 10, "hi": (i + 1) / 10, "n": 0, "pred_sum": 0.0,
             "actual_sum": 0.0} for i in range(10)]


def collect_mlb(results_dir, cal):
    """Mirrors loadMlbTrackRecord(): moneylines framed as "home team wins",
    props as over/under against their own line, Anytime HR as a plain yes/no."""
    store, dates = {}, []
    index = results_dir / "mlb_index.json"
    if not index.exists():
        return store, dates
    for date in json.loads(index.read_text(encoding="utf-8")).get("dates", []):
        f = results_dir / ("mlb_%s.json" % date)
        if not f.exists():
            continue
        day = json.loads(f.read_text(encoding="utf-8"))
        dates.append(date)
        for g in day.get("games") or []:
            if (g.get("already_played") and g.get("home_score") is not None
                    and g.get("away_score") is not None
                    and g.get("elo_home_prob") is not None):
                _record(store, "Moneyline", g["elo_home_prob"],
                        1 if g["home_score"] > g["away_score"] else 0, cal)
            for p in g.get("props") or []:
                if p.get("actual") is None:
                    continue
                is_hr = p.get("market") == "Anytime HR"
                prob = p.get("model_prob") if is_hr else p.get("model_over_prob")
                if prob is None:
                    continue
                if not is_hr and (p.get("line") is None or p["line"] <= 0):
                    continue
                over = (p["actual"] > 0) if is_hr else (p["actual"] > p["line"])
                _record(store, p["market"], prob, 1 if over else 0, cal)
    return store, dates


def collect_nfl(results_dir, cal):
    """NFL snapshots are one file per game, graded in place once played."""
    store, games = {}, 0
    market_n = market_correct = 0
    for f in sorted(results_dir.glob("nfl_*.json")):
        try:
            snap = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not snap.get("graded") or not snap.get("actual"):
            continue
        a = snap["actual"]
        games += 1
        if (snap.get("elo_home_prob") is not None
                and a.get("home_score") is not None and a.get("away_score") is not None):
            _record(store, "Moneyline", snap["elo_home_prob"],
                    1 if a["home_score"] > a["away_score"] else 0, cal)
        if a.get("market_pick"):
            market_n += 1
            market_correct += 1 if a.get("market_correct") else 0
        for p in snap.get("props_graded") or []:
            market = p.get("market") or "Prop"
            if market == "Anytime TD":
                # Binary, and its own probability field. hit is trustworthy
                # here because there is no line involved.
                if p.get("hit") is None or p.get("model_prob") is None:
                    continue
                _record(store, market, p["model_prob"], 1 if p["hit"] else 0, cal)
                continue

            # Everything else is an over/under, and the stored `hit` CANNOT be
            # used. grade_results grades a laddered prop against the middle
            # rung, while model_over_prob is the probability of clearing the
            # prop's own `line` -- 260 of 268 laddered props this week have
            # those two at different numbers (e.g. line 212.5, midpoint 210).
            # Scoring one against the other would put a probability for one
            # threshold next to an outcome for a different one. Recomputed
            # from actual_value against the base line instead, which is also
            # exactly how the MLB side scores its props.
            prob, line, actual = (p.get("model_over_prob"), p.get("line"),
                                  p.get("actual_value"))
            if prob is None or line is None or actual is None:
                continue
            _record(store, market, prob, 1 if actual > line else 0, cal)
    return store, games, market_n, market_correct


def rows_html(store):
    if not store:
        return ""
    out = []
    for market in sorted(store):
        b = store[market]
        n = b["n"]
        acc = b["correct"] / n
        rate = b["actual_sum"] / n
        base = max(rate, 1 - rate)
        brier = b["brier_sum"] / n
        delta = acc - base
        cls = "win" if delta > 0.02 else ("loss" if delta < -0.02 else "")
        out.append("<tr><td>%s</td><td class='num'>%s</td>"
                   "<td class='num %s'>%.0f%%</td><td class='num'>%.0f%%</td>"
                   "<td class='num'>%.4f</td></tr>"
                   % (e(market), f"{n:,}", cls, acc * 100, base * 100, brier))
    return "".join(out)


def table(title, store, note):
    if not store:
        return ""
    total_n = sum(b["n"] for b in store.values())
    total_c = sum(b["correct"] for b in store.values())
    return ("<h2>%s</h2><div class='card'><table><thead><tr><th>Market</th>"
            "<th>Graded</th><th>Accuracy</th><th>Baseline</th><th>Brier</th>"
            "</tr></thead><tbody>%s</tbody></table></div>"
            "<p class='note'>%s %s predictions graded, %.0f%% correct overall.</p>"
            % (e(title), rows_html(store), note, f"{total_n:,}", total_c / total_n * 100))


def calibration_table(cal):
    bins = [b for b in cal if b["n"] > 0]
    if len(bins) < 2:
        return ""
    rows = "".join(
        "<tr><td>%.0f%%&ndash;%.0f%%</td><td class='num'>%s</td>"
        "<td class='num'>%.1f%%</td><td class='num'>%.1f%%</td></tr>"
        % (b["lo"] * 100, b["hi"] * 100, f'{b["n"]:,}',
           b["pred_sum"] / b["n"] * 100, b["actual_sum"] / b["n"] * 100)
        for b in bins)
    return ("<h2>Calibration</h2><div class='card'><table><thead><tr>"
            "<th>Predicted range</th><th>Graded</th><th>We said</th>"
            "<th>Actually happened</th></tr></thead><tbody>%s</tbody></table></div>"
            "<p class='note'>The two right-hand columns should match. When we say "
            "something is 70%% likely, it should happen about 70%% of the time. A row "
            "where \"actually happened\" sits well below \"we said\" is the model "
            "being overconfident in that range, and that is worth more than any "
            "single accuracy number.</p>" % rows)


def build(docs_dir, results_dir):
    cal = new_calibration()
    mlb, mlb_dates = collect_mlb(results_dir, cal)
    nfl, nfl_games, mkt_n, mkt_correct = collect_nfl(results_dir, cal)

    total = sum(b["n"] for b in list(mlb.values()) + list(nfl.values()))
    canonical = "%s/track-record" % SITE
    title = "Track Record - How Accurate the Model Actually Is | Phins Up"
    desc = ("Every prediction Phins Up has made with a real result, graded in "
            "public: accuracy against a zero-skill baseline, Brier scores, and "
            "calibration. %s predictions and counting, including the ones we got "
            "wrong." % f"{total:,}")

    out = [head(title, desc, canonical)]
    out.append("<div class='crumb'><a href='%s/'>Phins Up</a></div>" % SITE)
    out.append("<h1>Track Record</h1>")
    out.append("<p class='kick'>Every prediction with a real result, including "
               "the ones we got wrong.</p>")

    out.append(
        "<div class='card'><p>Most sites in this category quote a win rate and "
        "show nothing behind it. This page is the whole record: <b>%s graded "
        "predictions</b> across every game moneyline and player prop the model "
        "has published, not a selected highlight reel, and not just the picks "
        "that made it onto a card.</p>"
        "<p class='note'>Two things to read it with. <b>Baseline</b> is the "
        "accuracy you would get with no skill at all, by always guessing "
        "whichever outcome happens more often -- an 89%% accuracy against an 89%% "
        "baseline is worth nothing, and we would rather show you that than hide "
        "it. <b>Brier score</b> measures how close probabilities land to reality: "
        "0.00 is perfect and 0.25 is what you score by calling everything a coin "
        "flip, so lower is better and anything meaningfully under 0.25 is doing "
        "real work.</p></div>" % f"{total:,}")

    if nfl:
        note = ("Graded from %d completed NFL games." % nfl_games)
        if mkt_n:
            note += (" On the same games the betting market called %.0f%% correctly."
                     % (mkt_correct / mkt_n * 100))
        out.append(table("NFL", nfl, note))
    else:
        out.append(
            "<h2>NFL</h2><div class='card'><p>No NFL games have been graded yet "
            "&mdash; the first results land after Week 1 is played. Predictions "
            "for every Week 1 game are already published and timestamped, so "
            "this table fills itself in without anything being chosen after the "
            "fact.</p></div>")

    if mlb:
        span = ("%s to %s" % (mlb_dates[0], mlb_dates[-1])) if mlb_dates else ""
        out.append(table("MLB", mlb,
                         "Graded from every finalized day, %s (%d days)."
                         % (span, len(mlb_dates))))

    out.append(calibration_table(cal))

    out.append(
        "<div class='cta'>These are statistical projections for information and "
        "entertainment, not betting advice, and a model that has been accurate is "
        "not a model that will be. The numbers above are here so you can judge "
        "that for yourself rather than take a claim on trust. "
        "<a href='%s/'>See this week's projections</a>.</div>" % SITE)
    out.append(FOOT)

    target = docs_dir / "track-record"
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text("".join(out), encoding="utf-8")
    print("Track record: %s graded predictions (NFL %d, MLB %d across %d days)"
          % (f"{total:,}", sum(b["n"] for b in nfl.values()),
             sum(b["n"] for b in mlb.values()), len(mlb_dates)))
    return canonical

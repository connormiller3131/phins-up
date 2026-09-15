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


# Markets no longer offered. Their history stays on this page -- deleting the
# rows would be exactly the selective editing this page exists to argue
# against -- but a reader has to be able to tell "we sell this and it is bad"
# apart from "we sold this, it was bad, we stopped".
WITHDRAWN = {
    "Pitcher Runs Allowed": "withdrawn Sep 2026",
    "Pitcher Outs Recorded": "withdrawn Sep 2026",
    "Pitcher Hits Allowed": "withdrawn Sep 2026",
}


# Below this many graded predictions a skill score is mostly noise, so it is
# shown without a verdict colour. Anytime TD swinging on twelve confident
# calls is the example that set this: the number is real, the certainty is not.
SKILL_MIN_N = 100


def skill_score(b):
    """Brier skill score: how much better the probabilities are than always
    quoting the base rate. 1 - brier / (p * (1-p)).

    This is the column that actually answers "is the model any good", and
    accuracy is not. Accuracy only asks which side of 50% a number fell on, so
    it is dominated by the base rate whenever one outcome is common: on week 1
    Anytime TD, 46 of 157 players scored, so saying "nobody scores" every time
    scores 70.7% and the model's 74.5% is four points of work, not three out of
    four. It misleads in the other direction too. Passing Yds came in at 69%
    accuracy against a 62% baseline and looked like the second best market on
    the board, while its skill score was -6.1%: it landed on the right side of
    50% more often than not, with probabilities worse than quoting the base
    rate. Returns None when the base rate is degenerate and there is nothing
    to score against."""
    n = b["n"]
    p = b["actual_sum"] / n
    no_skill = p * (1 - p)
    if no_skill <= 0:
        return None
    return 1 - (b["brier_sum"] / n) / no_skill


def _row_order(store):
    """Best markets first, then the ones there is not yet enough data to
    judge, then the failures. Alphabetical order buried the story: it put
    Anytime TD, the only market with real edge, in the middle of a list of
    markets showing none."""
    def key(market):
        b = store[market]
        bss = skill_score(b)
        if bss is None or b["n"] < SKILL_MIN_N:
            return (1, 0.0, market)      # unjudgeable, park in the middle
        return (0 if bss > 0 else 2, -bss, market)
    return sorted(store, key=key)


def rows_html(store):
    if not store:
        return ""
    out = []
    for market in _row_order(store):
        b = store[market]
        n = b["n"]
        acc = b["correct"] / n
        rate = b["actual_sum"] / n
        base = max(rate, 1 - rate)
        brier = b["brier_sum"] / n
        bss = skill_score(b)
        # The verdict colour lives on SKILL, not accuracy. It used to sit on
        # accuracy-minus-baseline, which painted Passing Yds green on a market
        # whose probabilities were worse than no model at all.
        if bss is None:
            skill_cls, skill_txt = "", "n/a"
        elif n < SKILL_MIN_N:
            # Named rather than merely uncoloured. A bare grey number still
            # reads as a verdict; "too early" says what it actually is.
            skill_cls, skill_txt = "", "%.1f%% <span class='tag'>too early</span>" % (bss * 100)
        else:
            skill_cls = "win" if bss > 0.01 else ("loss" if bss < -0.01 else "")
            skill_txt = "%.1f%%" % (bss * 100)
        label = e(market)
        if market in WITHDRAWN:
            label += " <span class='tag'>%s</span>" % e(WITHDRAWN[market])
        out.append("<tr><td>%s</td><td class='num'>%s</td>"
                   "<td class='num'>%.0f%%</td><td class='num'>%.0f%%</td>"
                   "<td class='num'>%.4f</td><td class='num %s'>%s</td></tr>"
                   % (label, f"{n:,}", acc * 100, base * 100, brier, skill_cls, skill_txt))
    return "".join(out)


def table(title, store, note):
    if not store:
        return ""
    total_n = sum(b["n"] for b in store.values())
    total_c = sum(b["correct"] for b in store.values())
    return ("<h2>%s</h2><div class='card'><table><thead><tr><th>Market</th>"
            "<th>Graded</th><th>Accuracy</th><th>Baseline</th><th>Brier</th><th>Skill</th>"
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
    # State the verdict rather than leaving a reader to diff two columns by
    # eye. Weighted by sample size, and the worst band named outright: a page
    # that exists to publish its own errors should not make you find them.
    tot = sum(b["n"] for b in bins)
    drift = sum((b["actual_sum"] - b["pred_sum"]) for b in bins) / tot
    worst = max(bins, key=lambda b: (b["pred_sum"] - b["actual_sum"]) / b["n"] * (b["n"] >= 100))
    worst_gap = (worst["pred_sum"] - worst["actual_sum"]) / worst["n"]
    if drift < -0.01:
        verdict = ("Across everything graded we run <b>%.1f points overconfident</b>: "
                   "things we call likely happen less often than we say." % (-drift * 100))
    elif drift > 0.01:
        verdict = ("Across everything graded we run <b>%.1f points underconfident</b>: "
                   "things we call likely happen more often than we say." % (drift * 100))
    else:
        verdict = "Across everything graded the two columns sit within a point of each other."
    if worst_gap > 0.02 and worst["n"] >= 100:
        verdict += (" The worst band is <b>%.0f%%&ndash;%.0f%%</b>, where we said %.1f%% and it "
                    "happened %.1f%% of the time across %s predictions."
                    % (worst["lo"] * 100, worst["hi"] * 100,
                       worst["pred_sum"] / worst["n"] * 100,
                       worst["actual_sum"] / worst["n"] * 100, f'{worst["n"]:,}'))
    return ("<h2>Calibration</h2><div class='card'><table><thead><tr>"
            "<th>Predicted range</th><th>Graded</th><th>We said</th>"
            "<th>Actually happened</th></tr></thead><tbody>%s</tbody></table></div>"
            "<p class='note'>The two right-hand columns should match. When we say "
            "something is 70%% likely, it should happen about 70%% of the time. %s "
            "That matters more than any single accuracy number, because it is the "
            "part you would actually be betting on.</p>" % (rows, verdict))


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
        "<p class='note'>How to read it. <b>Baseline</b> is the accuracy you "
        "would get with no skill at all, by always guessing whichever outcome "
        "happens more often -- an 89%% accuracy against an 89%% baseline is worth "
        "nothing, and we would rather show you that than hide it. "
        "<b>Brier score</b> measures how close probabilities land to reality: "
        "0.00 is perfect and 0.25 is what you score by calling everything a coin "
        "flip, so lower is better. <b>Skill</b> is the column to actually judge "
        "us on, and the only one here that a lopsided market cannot flatter. It "
        "asks whether our probabilities beat simply quoting how often the thing "
        "happens. Anytime TD is the example: 46 of 157 players scored, so "
        "\"nobody scores\" is already 71%% accurate and our 75%% is four points of "
        "work, not three-in-four. It catches the reverse too, where a market "
        "clears its accuracy baseline while its probabilities are worse than no "
        "model at all. Skill on fewer than 100 graded predictions is left "
        "uncoloured, because at that size it is mostly noise.</p></div>" % f"{total:,}")

    if nfl:
        note = ("Graded from %d completed NFL games." % nfl_games)
        if mkt_n:
            note += (" On the same games the betting market called %.0f%% correctly."
                     % (mkt_correct / mkt_n * 100))
        # The model against the market is the single most load-bearing number
        # on this page, and it was a clause at the end of a footnote. If the
        # market is beating us, that is the first thing a reader should see,
        # not something they find. Stated before the table, in both directions.
        ml = nfl.get("Moneyline")
        if mkt_n and ml and ml["n"]:
            ours = ml["correct"] / ml["n"] * 100
            theirs = mkt_correct / mkt_n * 100
            if theirs > ours + 0.5:
                head_line = ("<b>The betting market is currently beating us on NFL games.</b> "
                             "It has called %.0f%% of them correctly against our %.0f%%. "
                             "We publish that because a record you can only read when it "
                             "flatters us is not a record." % (theirs, ours))
            elif ours > theirs + 0.5:
                head_line = ("<b>We are currently ahead of the betting market on NFL games</b>, "
                             "%.0f%% to %.0f%%. On this sample size that is worth very little, "
                             "and we would rather say so now than pretend otherwise later."
                             % (ours, theirs))
            else:
                head_line = ("<b>We and the betting market are level on NFL games</b>, "
                             "%.0f%% against %.0f%%." % (ours, theirs))
            out.append("<div class='card'><p>%s</p></div>" % head_line)
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

    if any(m in WITHDRAWN for m in mlb):
        out.append(
            "<p class='note'><b>On the withdrawn markets.</b> Three pitcher props "
            "were pulled in September 2026 because of what this page showed: their "
            "Brier scores sat above 0.25, which is worse than calling every one a "
            "coin flip, so the probabilities were misleading rather than merely "
            "weak. Their history stays here rather than being deleted. They are "
            "not offered again until the models behind them are rebuilt.</p>")

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

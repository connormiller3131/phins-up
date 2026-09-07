"""Pre-render one static HTML page per NFL game, plus week and season index
pages, a sitemap and robots.txt.

WHY THIS EXISTS. The app is a single page: /nfl/week1, /mlb/2026-09-07 and /
all serve byte-identical HTML with one <title>, one description and a
canonical pointing at "/". Google can therefore index exactly one page, and
one page ranks for roughly one thing. On top of that every game number lives
in `const DATA = {...}` and only exists after JavaScript runs, which is the
least reliable way for a new domain to get its content seen.

These pages fix both: real content in the HTML, and a separate indexable URL
per game with its own title, description, canonical and heading.

SCOPED TO NFL ON PURPOSE. MLB would be ~94 pages a day and NHL similar. Mass
-producing thousands of near-identical daily pages is the doorway-page
pattern search engines demote sites for, and it would add tens of thousands
of files to the repo over a season. NFL is 16 games a week, the pages stay
useful for days rather than hours, and it is where the search volume is.

SAFE BY CONSTRUCTION. build_static_site calls this AFTER build_gated_payload
has already popped `props` off every game, so the game dicts reaching this
module physically do not contain paid data. Do not reorder those two calls.

DETERMINISTIC ON PURPOSE. Nothing here reads the clock except the sitemap
entries for pages that genuinely change daily. The pipeline reruns twice a
day; if these pages embedded a build timestamp, every rerun would rewrite
~300 files and the repo would carry a diff of pure noise. A game's page
changes only when that game's data does.
"""
import html
import json
import shutil

SITE = "https://phinsup.net"

# Mirrors dashboard_live.html's :root palette so a visitor landing here from
# a search result and then clicking through does not meet two different sites.
CSS = """
:root{--bg:#0E1A1C;--panel:#15262A;--panel-alt:#1B2E33;--line:#264047;
--text:#EDEFF2;--dim:#8FA6AB;--green:#00C2B8;--amber:#F5821F;--red:#F0555F;
--mono:'Consolas','SFMono-Regular',monospace;
--disp:'Century Gothic','Futura',-apple-system,'Segoe UI',sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:28px 20px 64px}
a{color:var(--green)}
.crumb{font-size:12.5px;color:var(--dim);margin-bottom:18px}
.crumb a{color:var(--dim);text-decoration:none}
.crumb a:hover{color:var(--green)}
h1{font-family:var(--disp);font-weight:800;font-size:30px;line-height:1.15;
margin:0 0 8px;letter-spacing:.01em}
.kick{color:var(--dim);font-size:13.5px;margin:0 0 22px}
h2{font-family:var(--disp);font-size:16px;letter-spacing:.06em;
text-transform:uppercase;color:var(--dim);margin:30px 0 10px;font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
tr:last-child td{border-bottom:none}
.num{font-family:var(--mono)}
.big{font-size:19px;font-weight:700}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;
padding:2px 7px;border-radius:4px;letter-spacing:.03em}
.good{background:rgba(0,194,184,.15);color:var(--green)}
.win{color:var(--green);font-weight:700}
.loss{color:var(--red);font-weight:700}
p.note{color:var(--dim);font-size:13px}
.cta{background:var(--panel-alt);border:1px solid var(--line);
border-radius:10px;padding:16px 18px;margin-top:26px}
.games a{display:flex;justify-content:space-between;gap:12px;padding:10px 12px;
border:1px solid var(--line);border-radius:8px;margin-bottom:7px;
text-decoration:none;color:var(--text);background:var(--panel)}
.games a:hover{border-color:var(--green)}
.games .m{color:var(--dim);font-family:var(--mono);font-size:12.5px}
footer{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);
color:var(--dim);font-size:12px}
"""

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def e(x):
    return html.escape(str(x), quote=True)


def nickname(full_name, abbr):
    """'New England Patriots' -> 'Patriots', 'San Francisco 49ers' -> '49ers'.

    Returns the word with the casing the league itself uses. Lowercasing and
    then .title()-ing it renders the 49ers as "49Ers", which is exactly the
    kind of detail that makes a page look auto-generated."""
    parts = (full_name or "").split()
    return parts[-1] if len(parts) > 1 else (abbr or "Team")


def slug_part(full_name, abbr):
    return nickname(full_name, abbr).lower()


def game_slug(g):
    return (slug_part(g.get("awayName"), g.get("awayAbbr")) + "-vs-"
            + slug_part(g.get("homeName"), g.get("homeAbbr")))


def pretty_date(iso):
    try:
        y, m, d = (int(v) for v in str(iso).split("-"))
        return "%s %d, %d" % (MONTHS[m - 1], d, y)
    except Exception:
        return iso or ""


def pct(x):
    return "-" if x is None else "%.1f%%" % (x * 100)


def odds(v):
    return "-" if v is None else ("+%d" % v if v > 0 else str(v))


def head(title, desc, canonical, ld=None):
    """Every page carries its OWN title, description and canonical. That is
    the entire reason these files exist -- the app has one of each, total."""
    if ld is None:
        ld = {
            "@context": "https://schema.org",
            "@type": "WebPage",
            "name": title,
            "description": desc,
            "url": canonical,
            "isPartOf": {"@type": "WebSite", "name": "Phins Up", "url": SITE + "/"},
        }
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%s</title>
<meta name="description" content="%s">
<link rel="canonical" href="%s">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0E1A1C">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Phins Up">
<meta property="og:title" content="%s">
<meta property="og:description" content="%s">
<meta property="og:url" content="%s">
<meta property="og:image" content="%s/og.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="%s">
<meta name="twitter:description" content="%s">
<meta name="twitter:image" content="%s/og.png">
<style>%s</style>
<script type="application/ld+json">%s</script>
</head>
<body><div class="wrap">""" % (
        e(title), e(desc), e(canonical), e(title), e(desc), e(canonical), SITE,
        e(title), e(desc), SITE, CSS, json.dumps(ld, separators=(",", ":")))


FOOT = """
<footer>
Phins Up provides statistical projections for informational and entertainment
purposes only. Nothing here is betting advice, and no projection guarantees an
outcome. Not affiliated with the NFL or any sportsbook.
&middot; <a href="%s/">phinsup.net</a>
</footer>
</div></body></html>
""" % SITE


def team_stats_table(stats, label):
    if not stats:
        return ""
    o = stats.get("offense") or {}
    d = stats.get("defense") or {}
    op, orr = o.get("passing") or {}, o.get("rushing") or {}
    dp, dr = d.get("passing") or {}, d.get("rushing") or {}
    rows = [
        ("Points / game", o.get("points_per_game"), None),
        ("Passing yds / game", op.get("yds_per_game"), dp.get("yds_per_game")),
        ("Rushing yds / game", orr.get("yds_per_game"), dr.get("yds_per_game")),
        ("Passing TD / game", op.get("td_per_game"), dp.get("td_per_game")),
        ("Rushing TD / game", orr.get("td_per_game"), dr.get("td_per_game")),
    ]
    body = "".join(
        "<tr><td>%s</td><td class='num'>%s</td><td class='num'>%s</td></tr>"
        % (e(n), "-" if a is None else a, "-" if b is None else b)
        for n, a, b in rows)
    return ("<div class='card'><table><thead><tr><th>%s</th><th>Offense</th>"
            "<th>Defense allows</th></tr></thead><tbody>%s</tbody></table></div>"
            % (e(label), body))


def game_page(g, season, week, result):
    away = g.get("awayName") or g["awayAbbr"]
    home = g.get("homeName") or g["homeAbbr"]
    an = nickname(away, g["awayAbbr"])
    hn = nickname(home, g["homeAbbr"])
    slug = game_slug(g)
    canonical = "%s/nfl/%s/week-%d/%s" % (SITE, season, week, slug)

    mh = g.get("market_home_prob")
    eh = g.get("elo_home_prob")
    ma = None if mh is None else 1 - mh
    ea = None if eh is None else 1 - eh

    title = "%s vs %s Prediction - NFL Week %d Model Odds | Phins Up" % (an, hn, week)

    # Built from this game's own numbers, so no two pages share a description.
    # A duplicated description across 16 pages would defeat the point of
    # splitting them up in the first place.
    if eh is not None and mh is not None:
        if eh >= 0.5:
            fav, fp, imp = hn, eh, mh
        else:
            fav, fp, imp = an, 1 - eh, 1 - mh
        desc = ("Our Elo model makes the %s %.1f%% to win %s at %s in NFL Week %d. "
                "The opening moneyline implies %.1f%%. Model vs market, the edge, "
                "and both teams' season stats." % (fav, fp * 100, an, hn, week, imp * 100))
    else:
        desc = ("%s at %s, NFL Week %d %s: model win probability, the real posted "
                "line, and both teams' season stats." % (an, hn, week, season))

    ld = {
        "@context": "https://schema.org",
        "@type": "SportsEvent",
        "name": "%s at %s" % (away, home),
        "sport": "American Football",
        "startDate": g.get("gameday"),
        "url": canonical,
        "homeTeam": {"@type": "SportsTeam", "name": home},
        "awayTeam": {"@type": "SportsTeam", "name": away},
    }
    if g.get("stadium"):
        ld["location"] = {"@type": "Place", "name": g["stadium"]}

    kick = pretty_date(g.get("gameday"))
    if g.get("weekday"):
        kick = "%s, %s" % (g["weekday"], kick)
    if g.get("gametime"):
        kick += " &middot; %s ET" % e(g["gametime"])
    if g.get("stadium"):
        kick += " &middot; %s" % e(g["stadium"])
        if g.get("roof"):
            kick += " (%s)" % e(g["roof"])

    gv_a = " <span class='tag good'>GOOD VALUE</span>" if g.get("good_value_away") else ""
    gv_h = " <span class='tag good'>GOOD VALUE</span>" if g.get("good_value_home") else ""

    out = [head(title, desc, canonical, ld)]
    out.append("<div class='crumb'><a href='%s/'>Phins Up</a> / "
               "<a href='%s/nfl/%s/'>NFL %s</a> / "
               "<a href='%s/nfl/%s/week-%d/'>Week %d</a></div>"
               % (SITE, SITE, season, season, SITE, season, week, week))
    out.append("<h1>%s vs %s &mdash; NFL Week %d Model Projection</h1>" % (e(an), e(hn), week))
    out.append("<p class='kick'>%s</p>" % kick)

    out.append("<h2>Model vs market</h2><div class='card'><table><thead><tr>"
               "<th>Team</th><th>Model win %</th><th>Fair win %</th><th>Edge</th>"
               "</tr></thead><tbody>")
    for name, ep, mp, gv in ((away, ea, ma, gv_a), (home, eh, mh, gv_h)):
        ed = None if (ep is None or mp is None) else ep - mp
        eds = "-" if ed is None else "%s%.1f%%" % ("+" if ed >= 0 else "", ed * 100)
        out.append("<tr><td>%s%s</td><td class='num big'>%s</td>"
                   "<td class='num'>%s</td><td class='num'>%s</td></tr>"
                   % (e(name), gv, pct(ep), pct(mp), eds))
    out.append("</tbody></table></div>")
    out.append(
        "<p class='note'><b>Model win %</b> is our own Elo rating, fit on real "
        "2019-2025 results and carried forward through every completed game "
        "since. It never looks at a betting line. <b>Fair win %</b> is what this "
        "game's real posted opening moneyline implies once the sportsbook's own "
        "margin is removed. <b>GOOD VALUE</b> means the model is higher than the "
        "market on that side &mdash; a disagreement, not a guarantee.</p>")

    if g.get("mlHome") is not None or g.get("spread_line") is not None:
        out.append("<h2>The posted line</h2><div class='card'><table><tbody>")
        if g.get("spread_line") is not None:
            out.append("<tr><td>Spread (home)</td><td class='num'>%s</td></tr>" % g["spread_line"])
        if g.get("total_line") is not None:
            out.append("<tr><td>Total</td><td class='num'>%s</td></tr>" % g["total_line"])
        out.append("<tr><td>%s moneyline</td><td class='num'>%s</td></tr>"
                   % (e(away), odds(g.get("mlAway"))))
        out.append("<tr><td>%s moneyline</td><td class='num'>%s</td></tr>"
                   % (e(home), odds(g.get("mlHome"))))
        out.append("</tbody></table></div>")

    a = (result or {}).get("actual")
    if (result or {}).get("graded") and a:
        cls = "win" if a.get("model_correct") else "loss"
        verdict = "correct" if a.get("model_correct") else "wrong"
        out.append("<h2>Result</h2><div class='card'>")
        out.append("<p class='big'>Final: %s %s &mdash; %s %s</p>"
                   % (e(g["awayAbbr"]), e(a.get("away_score")),
                      e(a.get("home_score")), e(g["homeAbbr"])))
        out.append("<p>The model picked <b>%s</b> &mdash; <span class='%s'>%s</span>."
                   % (e(a.get("model_pick")), cls, verdict))
        if a.get("market_pick"):
            out.append(" The market picked <b>%s</b> &mdash; %s."
                       % (e(a["market_pick"]), "correct" if a.get("market_correct") else "wrong"))
        out.append("</p><p class='note'>These are the numbers that were on screen "
                   "before kickoff. They are not rewritten after the fact.</p></div>")

    stats = (team_stats_table(g.get("awayTeamStats"), away)
             + team_stats_table(g.get("homeTeamStats"), home))
    if stats:
        out.append("<h2>Season stats</h2>" + stats)

    out.append(
        "<div class='cta'><b>Player props for this game</b> &mdash; projected "
        "passing, rushing and receiving yards, and Anytime TD graded against real "
        "DraftKings prices &mdash; are on the live site. The model's picks need a "
        "free account; the full prop tables are part of a subscription. "
        "<a href='%s/nfl/week%d'>Open %s vs %s on Phins Up</a>.</div>"
        % (SITE, week, e(an), e(hn)))
    out.append(FOOT)
    return "".join(out), slug, canonical


def week_index(season, week, games, is_current):
    canonical = "%s/nfl/%s/week-%d/" % (SITE, season, week)
    title = "NFL Week %d %s Model Predictions & Odds | Phins Up" % (week, season)
    desc = ("Model win probability against the real posted opening line for all %d "
            "NFL Week %d games, %s season. Independent Elo ratings, not a "
            "repackaged market price." % (len(games), week, season))
    out = [head(title, desc, canonical)]
    out.append("<div class='crumb'><a href='%s/'>Phins Up</a> / "
               "<a href='%s/nfl/%s/'>NFL %s</a></div>" % (SITE, SITE, season, season))
    out.append("<h1>NFL Week %d &mdash; %s Model Predictions</h1>" % (week, season))
    out.append("<p class='kick'>%d games &middot; %s</p>"
               % (len(games), "still to be played" if is_current else "completed"))
    out.append("<div class='games'>")
    for g in games:
        an = nickname(g.get("awayName"), g["awayAbbr"])
        hn = nickname(g.get("homeName"), g["homeAbbr"])
        eh = g.get("elo_home_prob")
        if eh is None:
            side = "-"
        elif eh >= 0.5:
            side = "%s %.0f%%" % (hn, eh * 100)
        else:
            side = "%s %.0f%%" % (an, (1 - eh) * 100)
        out.append("<a href='%s/nfl/%s/week-%d/%s'><span>%s vs %s</span>"
                   "<span class='m'>%s</span></a>"
                   % (SITE, season, week, game_slug(g), e(an), e(hn), e(side)))
    out.append("</div>")
    out.append(FOOT)
    return "".join(out)


def season_index(season, weeks):
    canonical = "%s/nfl/%s/" % (SITE, season)
    title = "NFL %s Model Predictions by Week | Phins Up" % season
    desc = ("Every published NFL week of the %s season: model win probability "
            "against the real posted line, game by game." % season)
    out = [head(title, desc, canonical)]
    out.append("<div class='crumb'><a href='%s/'>Phins Up</a></div>" % SITE)
    out.append("<h1>NFL %s &mdash; Model Predictions by Week</h1>" % season)
    out.append("<div class='games'>")
    for w, n in weeks:
        out.append("<a href='%s/nfl/%s/week-%d/'><span>Week %d</span>"
                   "<span class='m'>%d games</span></a>" % (SITE, season, w, w, n))
    out.append("</div>")
    out.append(FOOT)
    return "".join(out)


def build(nfl_data, docs_dir, results_dir, today_iso):
    season = nfl_data["season"]
    current = int(nfl_data["current_week"])
    nfl_root = docs_dir / "nfl" / str(season)
    # Rebuilt wholesale. A game that gets rescheduled, renamed or dropped
    # would otherwise leave an orphan page behind, still served, still
    # indexed, quoting numbers nothing in the pipeline produces any more.
    if nfl_root.exists():
        shutil.rmtree(nfl_root)

    urls = [("%s/" % SITE, today_iso)]
    weeks = []
    pages = 0

    for wk in sorted(nfl_data.get("weeks", {}), key=int):
        games = nfl_data["weeks"][wk].get("games") or []
        if not games:
            continue
        w = int(wk)
        wdir = nfl_root / ("week-%d" % w)
        wdir.mkdir(parents=True, exist_ok=True)
        for g in games:
            rp = results_dir / ("nfl_%s_wk%02d_%s_%s.json"
                                % (season, w, g["awayAbbr"], g["homeAbbr"]))
            result = None
            if rp.exists():
                try:
                    result = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    result = None
            page, slug, canonical = game_page(g, season, w, result)
            gdir = wdir / slug
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / "index.html").write_text(page, encoding="utf-8")
            # lastmod is the game's own date, not today's: a finished game's
            # page never changes again, and telling Google otherwise twice a
            # day teaches it to ignore the signal.
            urls.append((canonical, g.get("gameday") or today_iso))
            pages += 1
        (wdir / "index.html").write_text(
            week_index(season, w, games, w == current), encoding="utf-8")
        urls.append(("%s/nfl/%s/week-%d/" % (SITE, season, w),
                     today_iso if w == current else (games[-1].get("gameday") or today_iso)))
        weeks.append((w, len(games)))

    nfl_root.mkdir(parents=True, exist_ok=True)
    (nfl_root / "index.html").write_text(season_index(season, weeks), encoding="utf-8")
    urls.append(("%s/nfl/%s/" % (SITE, season), today_iso))

    body = "".join("  <url><loc>%s</loc><lastmod>%s</lastmod></url>\n"
                   % (html.escape(u, quote=True), m) for u, m in urls)
    (docs_dir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + body + "</urlset>\n", encoding="utf-8")
    (docs_dir / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nDisallow: /api/\n\n"
        "Sitemap: %s/sitemap.xml\n" % SITE, encoding="utf-8")

    print("Game pages: %d games across %d week(s), %d URLs in sitemap.xml"
          % (pages, len(weeks), len(urls)))
    return pages

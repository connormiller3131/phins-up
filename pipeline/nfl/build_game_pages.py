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

NFL ONLY, for now, via BUILD_MLB_PAGES below. The MLB builder is written and
works; it is switched off because of proportion, not quality -- see the note
on that flag. The short version: 185 MLB URLs against 18 NFL meant almost all
of a new domain's crawl budget went to games with a six-hour relevance window.

(An earlier version of this note blamed doorway-page penalties and repo size.
Both were overstated. These pages carry real per-game data and plenty of
established sites publish one page per fixture; the repo cost is real but
manageable at ~8 KB a page. Crawl budget was the argument that actually
decided it.)

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

# MLB per-game pages are OFF.
#
# Not because they are wrong. They are built, they work, and the content is
# real -- probable pitchers, model vs market, final scores. The problem is
# PROPORTION. With them on the sitemap was 185 MLB URLs against 18 NFL, so
# ~91% of a brand-new domain's small crawl allocation went to baseball games
# with a six-hour relevance window, in the middle of football season. That is
# budget not being spent on the NFL pages, which are the ones with a realistic
# chance of ranking. MLB was not neutral here, it was crowding them out.
#
# Switched off while nothing MLB had been indexed yet, which is exactly what
# made removal free: no 404s on pages Google had already decided to keep. The
# same change in a month would have cost something.
#
# Turn back on (set True) when BOTH are true:
#   - the NFL pages are indexed and drawing impressions in Search Console
#   - the domain has a few months of history, so ~94 new pages a day reads as
#     a growing site rather than a spike
# Spring training is the natural moment.
BUILD_MLB_PAGES = False

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
&middot; <a href="https://x.com/PhinsUpDotNet" rel="noopener">@PhinsUpDotNet</a>
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
        # Name the favourite and the OTHER team -- an earlier version read
        # "the Dolphins 57.9% to win Dolphins at Raiders", repeating the away
        # side, which is exactly the tell that a description was generated.
        if eh >= 0.5:
            fav, dog, fp, imp = hn, an, eh, mh
        else:
            fav, dog, fp, imp = an, hn, 1 - eh, 1 - mh
        desc = ("Our Elo model makes the %s %.1f%% to beat the %s in NFL Week %d. "
                "The opening moneyline implies %.1f%%. Model vs market, the edge, "
                "and both teams' season stats." % (fav, fp * 100, dog, week, imp * 100))
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


# ---------------------------------------------------------------- MLB ------
# The NFL rule (last word of the team name) is WRONG here: "Boston Red Sox"
# and "Chicago White Sox" both end in "Sox", so they would collide on one slug
# and overwrite each other's page. Strip the city instead and keep the whole
# nickname. Longest prefix wins, so "Kansas City" is not read as "Kansas".
MLB_CITIES = [
    "Arizona", "Atlanta", "Baltimore", "Boston", "Chicago", "Cincinnati",
    "Cleveland", "Colorado", "Detroit", "Houston", "Kansas City",
    "Los Angeles", "Miami", "Milwaukee", "Minnesota", "New York", "Oakland",
    "Philadelphia", "Pittsburgh", "San Diego", "San Francisco", "Seattle",
    "St. Louis", "Tampa Bay", "Texas", "Toronto", "Washington",
]


def mlb_nickname(full_name, abbr):
    """'Boston Red Sox' -> 'Red Sox'. 'Athletics' -> 'Athletics'."""
    name = (full_name or "").strip()
    if not name:
        return abbr or "Team"
    for city in sorted(MLB_CITIES, key=len, reverse=True):
        if name.startswith(city + " "):
            return name[len(city) + 1:]
    return name


def slugify(text):
    out = []
    for ch in (text or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_" and out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-") or "team"


def mlb_game_slug(g):
    return (slugify(mlb_nickname(g.get("awayName"), g.get("awayAbbr"))) + "-vs-"
            + slugify(mlb_nickname(g.get("homeName"), g.get("homeAbbr"))))


def mlb_team_stats_table(stats, label):
    if not stats:
        return ""
    bat = ((stats.get("offense") or {}).get("batting")) or {}
    pit = ((stats.get("defense") or {}).get("pitching")) or {}
    rows = [
        ("Hits / game", bat.get("hits_per_game")),
        ("Total bases / game", bat.get("total_bases_per_game")),
        ("Home runs / game", bat.get("hr_per_game")),
        ("Walks / game", bat.get("bb_per_game")),
        ("ERA", pit.get("era")),
        ("Runs allowed / game", pit.get("runs_allowed_per_game")),
        ("Hits allowed / game", pit.get("hits_allowed_per_game")),
    ]
    body = "".join("<tr><td>%s</td><td class='num'>%s</td></tr>"
                   % (e(n), "-" if v is None else v) for n, v in rows)
    return ("<div class='card'><table><thead><tr><th>%s</th><th>Season</th></tr>"
            "</thead><tbody>%s</tbody></table></div>" % (e(label), body))


def mlb_game_page(g, date, slug):
    away = g.get("awayName") or g["awayAbbr"]
    home = g.get("homeName") or g["homeAbbr"]
    an = mlb_nickname(away, g["awayAbbr"])
    hn = mlb_nickname(home, g["homeAbbr"])
    canonical = "%s/mlb/%s/%s" % (SITE, date, slug)
    nice_date = pretty_date(date)

    mkt = g.get("market") or {}
    mh = mkt.get("home_fair_prob")
    eh = g.get("elo_home_prob")
    ma = None if mh is None else 1 - mh
    ea = None if eh is None else 1 - eh

    title = "%s vs %s Prediction - MLB %s Model Odds | Phins Up" % (an, hn, nice_date)

    if eh is not None and mh is not None:
        if eh >= 0.5:
            fav, dog, fp, imp = hn, an, eh, mh
        else:
            fav, dog, fp, imp = an, hn, 1 - eh, 1 - mh
        desc = ("Our model makes the %s %.1f%% to beat the %s on %s. The "
                "DraftKings moneyline implies %.1f%%. Probable pitchers, model vs "
                "market, and both teams' season stats."
                % (fav, fp * 100, dog, nice_date, imp * 100))
    else:
        desc = ("%s at %s on %s: model win probability, probable pitchers, and "
                "both teams' season stats." % (an, hn, nice_date))

    ld = {
        "@context": "https://schema.org", "@type": "SportsEvent",
        "name": "%s at %s" % (away, home), "sport": "Baseball",
        "startDate": g.get("gameDatetime") or date, "url": canonical,
        "homeTeam": {"@type": "SportsTeam", "name": home},
        "awayTeam": {"@type": "SportsTeam", "name": away},
    }

    out = [head(title, desc, canonical, ld)]
    out.append("<div class='crumb'><a href='%s/'>Phins Up</a> / "
               "<a href='%s/mlb/%s'>MLB &middot; %s</a></div>"
               % (SITE, SITE, date, e(nice_date)))
    out.append("<h1>%s vs %s &mdash; MLB Model Projection</h1>" % (e(an), e(hn)))
    out.append("<p class='kick'>%s</p>" % e(nice_date))

    ap, hp = g.get("awayProbablePitcher"), g.get("homeProbablePitcher")
    if ap or hp:
        out.append("<h2>Probable pitchers</h2><div class='card'><table><tbody>"
                   "<tr><td>%s</td><td>%s</td></tr><tr><td>%s</td><td>%s</td></tr>"
                   "</tbody></table></div>"
                   % (e(away), e(ap or "not announced"),
                      e(home), e(hp or "not announced")))

    out.append("<h2>Model vs market</h2><div class='card'><table><thead><tr>"
               "<th>Team</th><th>Model win %</th><th>Fair win %</th><th>Edge</th>"
               "</tr></thead><tbody>")
    gv_a = " <span class='tag good'>GOOD VALUE</span>" if g.get("good_value_away") else ""
    gv_h = " <span class='tag good'>GOOD VALUE</span>" if g.get("good_value_home") else ""
    for name, ep, mp, gv in ((away, ea, ma, gv_a), (home, eh, mh, gv_h)):
        ed = None if (ep is None or mp is None) else ep - mp
        eds = "-" if ed is None else "%s%.1f%%" % ("+" if ed >= 0 else "", ed * 100)
        out.append("<tr><td>%s%s</td><td class='num big'>%s</td>"
                   "<td class='num'>%s</td><td class='num'>%s</td></tr>"
                   % (e(name), gv, pct(ep), pct(mp), eds))
    out.append("</tbody></table></div>")
    out.append(
        "<p class='note'><b>Model win %</b> is our own rating -- an Elo fit on real "
        "2019-2025 results, refined by who is actually pitching, the bullpen behind "
        "him and how the lineup has been hitting. It never looks at a betting line. "
        "<b>Fair win %</b> is what DraftKings' moneyline implies once the book's own "
        "margin is removed. <b>GOOD VALUE</b> means the model is higher than the "
        "market on that side &mdash; a disagreement, not a guarantee.</p>")

    if mkt.get("mlHome") is not None or mkt.get("total_line") is not None:
        out.append("<h2>The posted line</h2><div class='card'><table><tbody>")
        if mkt.get("run_line_home") is not None:
            out.append("<tr><td>Run line (home)</td><td class='num'>%s</td></tr>"
                       % mkt["run_line_home"])
        if mkt.get("total_line") is not None:
            out.append("<tr><td>Total</td><td class='num'>%s</td></tr>" % mkt["total_line"])
        out.append("<tr><td>%s moneyline</td><td class='num'>%s</td></tr>"
                   % (e(away), odds(mkt.get("mlAway"))))
        out.append("<tr><td>%s moneyline</td><td class='num'>%s</td></tr>"
                   % (e(home), odds(mkt.get("mlHome"))))
        out.append("</tbody></table></div>")

    if g.get("already_played") and g.get("home_score") is not None:
        aw, hs = g.get("away_score"), g.get("home_score")
        model_home = eh is not None and eh >= 0.5
        correct = None if (eh is None or hs == aw) else ((hs > aw) == model_home)
        out.append("<h2>Result</h2><div class='card'>")
        out.append("<p class='big'>Final: %s %s &mdash; %s %s</p>"
                   % (e(g["awayAbbr"]), e(aw), e(hs), e(g["homeAbbr"])))
        if correct is not None:
            out.append("<p>The model favoured <b>%s</b> &mdash; <span class='%s'>%s</span>.</p>"
                       % (e(home if model_home else away),
                          "win" if correct else "loss",
                          "correct" if correct else "wrong"))
        out.append("<p class='note'>These are the numbers that were on screen "
                   "before first pitch. They are not rewritten afterwards.</p></div>")

    stats = (mlb_team_stats_table(g.get("awayTeamStats"), away)
             + mlb_team_stats_table(g.get("homeTeamStats"), home))
    if stats:
        out.append("<h2>Season stats</h2>" + stats)

    out.append(
        "<div class='cta'><b>Player props for this game</b> &mdash; every hitter on "
        "the active roster plus the probable starter, with Hits, Total Bases, RBI, "
        "strikeouts and Anytime HR &mdash; are on the live site. The model's picks "
        "need a free account; the full prop tables are part of a subscription. "
        "<a href='%s/mlb/%s'>Open %s vs %s on Phins Up</a>.</div>"
        % (SITE, date, e(an), e(hn)))
    out.append(FOOT)
    return "".join(out)


def build_mlb(mlb_data, docs_dir):
    """Write a page per game for every day in the current slate payload.

    Deliberately does NOT wipe docs/mlb wholesale the way the NFL tree is
    rebuilt. The slate is a rolling ~7-day window, so wiping would delete the
    page for every game older than a week -- finished games whose pages are
    correct, permanent, and possibly already indexed. Only days present in
    this payload are rebuilt; anything older is left untouched.

    There is also NO day-index page, and there must not be: /mlb/<date> is a
    live client-side route of the app, so writing docs/mlb/<date>/index.html
    would shadow it and replace that day's whole slate with one static page.
    """
    built = []
    for date, day in sorted((mlb_data.get("days") or {}).items()):
        games = day.get("games") or []
        if not games:
            continue
        ddir = docs_dir / "mlb" / date
        if ddir.exists():
            shutil.rmtree(ddir)
        ddir.mkdir(parents=True, exist_ok=True)
        seen = {}
        for g in games:
            slug = mlb_game_slug(g)
            # Doubleheaders are two games between the same teams on the same
            # date. Without this the second silently overwrites the first.
            seen[slug] = seen.get(slug, 0) + 1
            if seen[slug] > 1:
                slug = "%s-game-%d" % (slug, seen[slug])
            gdir = ddir / slug
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / "index.html").write_text(mlb_game_page(g, date, slug), encoding="utf-8")
            g["page_url"] = "/mlb/%s/%s" % (date, slug)
            built.append(("%s/mlb/%s/%s" % (SITE, date, slug), date))
    return built


def build(nfl_data, mlb_data, docs_dir, results_dir, today_iso):
    season = nfl_data["season"]
    current = int(nfl_data["current_week"])
    nfl_root = docs_dir / "nfl" / str(season)
    # Rebuilt wholesale. A game that gets rescheduled, renamed or dropped
    # would otherwise leave an orphan page behind, still served, still
    # indexed, quoting numbers nothing in the pipeline produces any more.
    if nfl_root.exists():
        shutil.rmtree(nfl_root)

    urls = [("%s/" % SITE, today_iso), ("%s/track-record" % SITE, today_iso)]
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
            # Stamped onto the game itself so the page can link to it by
            # reading a field, instead of reimplementing the slug rule in
            # JavaScript -- which for MLB would mean duplicating the city
            # -stripping list too, and getting the Red Sox wrong if it drifted.
            g["page_url"] = "/nfl/%s/week-%d/%s" % (season, w, slug)
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

    if BUILD_MLB_PAGES:
        mlb_built = build_mlb(mlb_data, docs_dir)
    else:
        # Authoritative, not merely inert: with the flag off the tree is
        # REMOVED, so a stale working copy or an old CI cache cannot quietly
        # keep serving pages the sitemap no longer lists. Both workflows stage
        # docs/mlb with -A, so the deletions are committed rather than left
        # behind on the deployed site.
        mlb_built = []
        if (docs_dir / "mlb").exists():
            shutil.rmtree(docs_dir / "mlb")

    # The sitemap is built by SCANNING what is on disk, not from the list this
    # run happened to write. MLB keeps a rolling ~7-day payload but its pages
    # are never deleted, so days that dropped out of the slate still exist and
    # still belong in the sitemap -- listing only this run's output would
    # silently drop the whole archive every day. The date is read back out of
    # the URL, which is why it is a path segment.
    mlb_urls = []
    mlb_root = docs_dir / "mlb"
    if mlb_root.exists():
        for f in sorted(mlb_root.glob("*/*/index.html")):
            date, slug = f.parent.parent.name, f.parent.name
            mlb_urls.append(("%s/mlb/%s/%s" % (SITE, date, slug), date))
    urls.extend(mlb_urls)

    body = "".join("  <url><loc>%s</loc><lastmod>%s</lastmod></url>\n"
                   % (html.escape(u, quote=True), m) for u, m in urls)
    (docs_dir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + body + "</urlset>\n", encoding="utf-8")
    (docs_dir / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nDisallow: /api/\n\n"
        "Sitemap: %s/sitemap.xml\n" % SITE, encoding="utf-8")

    print("Game pages: NFL %d games across %d week(s); MLB %d games this run, "
          "%d on disk; %d URLs in sitemap.xml"
          % (pages, len(weeks), len(mlb_built), len(mlb_urls), len(urls)))
    return pages

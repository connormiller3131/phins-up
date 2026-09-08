"""Real DraftKings moneylines for NHL games, from ESPN's public scoreboard.

WHY ESPN AND NOT THE ODDS API. The Odds API bills per call against a monthly
credit budget -- tight enough that MLB's prop pull is capped at two games a
run. ESPN's scoreboard carries DraftKings prices for free, with no key, and it
returns BOTH the opening and the current line, which is exactly the pair the
NFL tab already shows as "Fair win % (pregame)" and "(current)". Verified
against a real slate: 8 of 8 games had complete open and close moneylines, and
the provider really is DraftKings rather than ESPN BET, so the site's existing
copy stays true.

THE ONE CONSTRAINT: odds only exist on UPCOMING games. ESPN strips the block
once a game is final, so a line has to be captured before puck drop and stored
-- it cannot be backfilled afterwards. Attaching on every run, as the NFL side
does with its own current lines, is what makes that work.

FAILS SOFT AT EVERY STAGE. A network error, a shape change, or a missing game
leaves the affected fields unset rather than breaking the run. The NHL tab
worked with no market odds at all before this and must keep working if ESPN
goes away.
"""
import json
import urllib.error
import urllib.request

SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/"
              "scoreboard?dates=%s&limit=100")

# The site uses NHL's own abbreviations; ESPN uses its own. Deliberately NOT
# shared with dashboard_live.html's ESPN_ABBR_RENAME: that map is for ESPN's
# BOX SCORE endpoint, and the two endpoints disagree. The box-score map sends
# UTA to "UTAH", while the scoreboard returns plain "UTA" -- reusing it would
# silently drop Utah's odds for a whole season. Verified against a real
# scoreboard response; add to this map only after checking THIS endpoint.
SITE_TO_ESPN = {"NJD": "NJ", "SJS": "SJ", "TBL": "TB", "LAK": "LA"}


def _implied_prob(american):
    a = float(american)
    return (-a) / (-a + 100) if a < 0 else 100 / (a + 100)


def _odds_int(node, side, phase):
    """odds.moneyline.<side>.<phase>.odds, as an int, or None.

    ESPN returns these as strings like "-110" / "+164", and any level of the
    chain can be missing on a game the book has not priced.
    """
    try:
        raw = node[side][phase]["odds"]
    except (KeyError, TypeError):
        return None
    try:
        return int(str(raw).replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def fetch_day(date_iso):
    """{(awayEspn, homeEspn): {...}} for one YYYY-MM-DD. Empty on any failure."""
    url = SCOREBOARD % date_iso.replace("-", "")
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            data = json.load(r)
    except (urllib.error.URLError, ValueError, TimeoutError, OSError) as e:
        print(f"[nhl_odds] {date_iso}: scoreboard fetch failed ({e}); no odds this day")
        return {}

    out = {}
    for event in data.get("events") or []:
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        odds = (comp.get("odds") or [None])[0]
        if not odds or not odds.get("moneyline"):
            continue
        ml = odds["moneyline"]
        home = away = None
        for c in comp.get("competitors") or []:
            abbr = (c.get("team") or {}).get("abbreviation")
            if c.get("homeAway") == "home":
                home = abbr
            elif c.get("homeAway") == "away":
                away = abbr
        if not home or not away:
            continue
        out[(away, home)] = {
            "open_away": _odds_int(ml, "away", "open"),
            "open_home": _odds_int(ml, "home", "open"),
            "close_away": _odds_int(ml, "away", "close"),
            "close_home": _odds_int(ml, "home", "close"),
            "total": odds.get("overUnder"),
            "provider": (odds.get("provider") or {}).get("name"),
        }
    return out


def _no_vig_home(away_ml, home_ml):
    pa, ph = _implied_prob(away_ml), _implied_prob(home_ml)
    total = pa + ph
    return round(ph / total, 4) if total else None


def attach(games_by_date):
    """Attach moneylines to {date: [game, ...]}, mutating in place.

    Mirrors the NFL field names exactly, because the NHL card reuses that
    template's markup: mlAway/mlHome + market_home_prob for the opening line,
    current_* for the live one, and good_value_* for the disagreement.
    """
    attached = days = 0
    for date_iso, games in sorted(games_by_date.items()):
        if not games:
            continue
        odds_map = fetch_day(date_iso)
        if not odds_map:
            continue
        days += 1
        for g in games:
            key = (SITE_TO_ESPN.get(g["awayAbbr"], g["awayAbbr"]),
                   SITE_TO_ESPN.get(g["homeAbbr"], g["homeAbbr"]))
            m = odds_map.get(key)
            if not m:
                continue

            if m["open_away"] is not None and m["open_home"] is not None:
                g["mlAway"] = m["open_away"]
                g["mlHome"] = m["open_home"]
                g["market_home_prob"] = _no_vig_home(m["open_away"], m["open_home"])
            if m["close_away"] is not None and m["close_home"] is not None:
                g["current_mlAway"] = m["close_away"]
                g["current_mlHome"] = m["close_home"]
                g["current_market_home_prob"] = _no_vig_home(m["close_away"], m["close_home"])
            if m.get("total") is not None:
                g["total_line"] = m["total"]

            # GOOD VALUE is judged against the OPENING line, the same as NFL:
            # it should reflect the model's read at the moment the price was
            # set rather than chase wherever the market has since moved.
            mh = g.get("market_home_prob")
            eh = g.get("elo_home_prob")
            if mh is not None and eh is not None:
                g["good_value_home"] = bool(eh > mh)
                g["good_value_away"] = bool((1 - eh) > (1 - mh))
            if g.get("mlHome") is not None or g.get("current_mlHome") is not None:
                attached += 1

    print(f"[nhl_odds] attached DraftKings moneylines to {attached} game(s) "
          f"across {days} day(s)")
    return attached

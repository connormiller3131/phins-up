"""One theme for every page the Python build writes: game pages, the track
record, and the week and season indexes.

The main app (pipeline/nfl/dashboard_live.html) carries the same values in its
own :root block, because it is a hand-written template rather than a build
output. If a color changes here, change it there too.

THE PALETTE is the Miami Dolphins' own: aqua, navy and orange, on a light
ground by default with a dark mode that stays Dolphins (deep navy) rather than
going back to black and teal.

ONE DELIBERATE DEPARTURE FROM THE BRAND HEX. Dolphins aqua #008e97 measures
3.95:1 as text on white, under the 4.5:1 accessibility minimum, so text and the
active tab use #007a82 (4.9:1), which still reads as the same aqua. Measured on
the rendered pages in both themes; every text element clears 4.5:1.

TOKEN NAMES. The game pages were written with --surface/--dim/--accent and the
older pages with --panel/--text-dim/--green. Both sets are defined here as
aliases of the same values, so neither needed rewriting to adopt the theme.
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700'
         '&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">')

# Applied in <head>, before first paint, so a dark-mode visitor never sees a
# flash of the light page. Shares the "pu-theme" key with the main app, so the
# choice follows you across the whole site.
HEAD_SCRIPT = ('<script>try{if(localStorage.getItem("pu-theme")==="dark")'
               'document.documentElement.setAttribute("data-theme","dark")}catch(e){}</script>')

TOKENS = """
:root{
--bg:#f3f6f7;--surface:#ffffff;--panel:#ffffff;--surface2:#eaf2f4;--panel-alt:#eaf2f4;
--line:#d6e3e7;--row:#e6eef1;--text:#0f2a33;--dim:#4d6873;--text-dim:#4d6873;--faint:#567280;
--accent:#007a82;--green:#007a82;--amber:#d9480f;--red:#c92a2a;--live:#e03131;
--accent-tint:rgba(0,122,130,.10);--bad-tint:rgba(201,42,42,.10);--on-accent:#ffffff;
--nav-bg:#005778;--nav-text:#ffffff;--nav-dim:#a9cdd8;--nav-active:#007a82;--brand-orange:#fc4c02;
--scroll:#c4d6db;--swatch:#9bb3ba;--chart-line:#0f2a33;--chart-mid:#c4d6db;
--disp:'Oswald','Arial Narrow',system-ui,sans-serif;
--body:'Source Sans 3',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
--mono:'Source Sans 3',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
html[data-theme="dark"]{
--bg:#07161c;--surface:#0e232b;--panel:#0e232b;--surface2:#13303a;--panel-alt:#13303a;
--line:#1f3d47;--row:#18333c;--text:#e8f1f3;--dim:#8fb0b8;--text-dim:#8fb0b8;--faint:#7496a0;
--accent:#1ab5bf;--green:#1ab5bf;--amber:#ff7a3d;--red:#ff6b6b;--live:#ff5a5f;
--accent-tint:rgba(26,181,191,.14);--bad-tint:rgba(255,107,107,.14);--on-accent:#041016;
--nav-bg:#041016;--nav-text:#e8f1f3;--nav-dim:#7fa3ad;--nav-active:#005778;--brand-orange:#ff6a2b;
--scroll:#2a4a54;--swatch:#4d6b74;--chart-line:#e8f1f3;--chart-mid:#2a4a54}
.theme-toggle{display:flex;align-items:center;justify-content:center;width:34px;height:34px;
border-radius:6px;border:1px solid rgba(255,255,255,.18);background:none;color:var(--nav-text);cursor:pointer}
.theme-toggle:hover{background:rgba(255,255,255,.08)}
.theme-toggle svg{width:17px;height:17px}
.theme-toggle .sun{display:none}
html[data-theme="dark"] .theme-toggle .sun{display:block}
html[data-theme="dark"] .theme-toggle .moon{display:none}
"""

NAV_CSS = """
.nav{display:flex;align-items:center;gap:22px;height:52px;padding:0 20px;
background:var(--nav-bg);position:sticky;top:0;z-index:20}
.brand{font:700 21px/1 var(--disp);letter-spacing:.04em;color:var(--nav-text)}
.brand b{color:var(--brand-orange);font-weight:700}
.nav .sp{display:flex;gap:4px}
.nav .sp a{font:600 15px/1 var(--disp);letter-spacing:.06em;color:var(--nav-dim);
padding:8px 10px;border-radius:6px;text-decoration:none}
.nav .sp a:hover{color:var(--nav-text)}
.nav .sp a.on{color:#fff;background:var(--nav-active)}
.nav .rt{margin-left:auto;display:flex;align-items:center;gap:18px;font-size:13px}
.nav .rt a{color:var(--nav-dim);text-decoration:none}
.nav .rt a:hover{color:var(--nav-text)}
"""

TOGGLE_BUTTON = (
    '<button class="theme-toggle" type="button" onclick="toggleTheme()" '
    'aria-label="Toggle dark mode" title="Dark mode">'
    '<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
    '<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/>'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '</button>')

TOGGLE_JS = ('<script>function toggleTheme(){var r=document.documentElement,'
             'd=r.getAttribute("data-theme")!=="dark";'
             'if(d)r.setAttribute("data-theme","dark");else r.removeAttribute("data-theme");'
             'try{localStorage.setItem("pu-theme",d?"dark":"light")}catch(e){}'
             'if(window.onThemeChange)window.onThemeChange()}</script>')


def nav(site, active="nfl"):
    """The same bar the app shows: wordmark, sports, track record, toggle."""
    sp = "".join('<a href="%s/%s" class="%s">%s</a>' % (site, h, "on" if k == active else "", lab)
                 for k, h, lab in (("nfl", "", "NFL"), ("mlb", "mlb", "MLB"), ("nhl", "nhl", "NHL")))
    return ('<header class="nav"><a class="brand" href="%s/">PHINS <b>UP</b></a>'
            '<nav class="sp">%s</nav><div class="rt"><a href="%s/track-record">Track record</a>'
            '%s</div></header>' % (site, sp, site, TOGGLE_BUTTON))

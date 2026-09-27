"""NFL game pages, built like a real sports site rather than a report.

WHAT CHANGED AND WHY. The old page opened with a sentence for a headline, a
table, and a paragraph explaining the table. Sports sites lead with the game:
who is playing, the score, the clock, then the numbers. This page does that,
and pushes our own model into its own section instead of wrapping every
number in a methodology note.

TWO LAYERS.
  Static, rendered here at build time: the matchup, date, venue, our model
  block, the posted line, the result and season stats. This is what search
  engines index and what shows if ESPN is slow or down.
  Live, fetched in the browser from ESPN's public site API (the same host the
  app already uses for box scores, which sends Access-Control-Allow-Origin: *):
  the scoreboard strip, live score and clock, linescore, box score, team
  stats, scoring plays, win probability and pregame injuries. It refreshes
  every 30 seconds while the game is live and stops when it is not.

THE PAYWALL HOLDS. The model's own win probability is rendered only for the
three free primetime games, exactly as before. Every live section is public
ESPN data and never touches the model.

NO LOGOS. ESPN returns them, but they are team trademarks and this is a paid
product, so the look comes from each team's own colors instead. Two teams
that share a primary (Seattle and New England are both navy) fall back to the
alternate color so the header never shows the same color twice.
"""
import json

from pipeline.nfl.build_game_pages import (SITE, e, nickname, pct, odds, pretty_date,
                                           team_stats_table)
from pipeline import site_theme

# ESPN spells two NFL teams differently from nflverse.
ESPN_ABBR = {"LA": "LAR", "WAS": "WSH"}

FONTS = site_theme.FONTS

CSS = """
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 var(--body);
-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
a{color:inherit;text-decoration:none}
.num{font-variant-numeric:tabular-nums}

/* scoreboard strip */
.strip{display:flex;gap:1px;overflow-x:auto;background:var(--line);
border-bottom:1px solid var(--line);scrollbar-width:thin;scrollbar-color:var(--scroll) transparent}
.strip::-webkit-scrollbar{height:6px}.strip::-webkit-scrollbar-track{background:transparent}
.strip::-webkit-scrollbar-thumb{background:var(--scroll);border-radius:3px}
.tile{flex:0 0 150px;background:var(--surface);padding:9px 12px 10px;display:block}
.tile:hover{background:var(--surface2)}
.tile.cur{background:var(--surface2);box-shadow:inset 0 -2px 0 var(--accent)}
.tile .st{font:600 11px/1 var(--body);color:var(--dim);letter-spacing:.03em;
text-transform:uppercase;margin-bottom:7px;display:flex;align-items:center;gap:6px}
.tile .st.live{color:var(--live)}
.tile .st.live:before{content:"";width:6px;height:6px;border-radius:50%;background:var(--live)}
.tile .tr{display:flex;justify-content:space-between;font:600 15px/1.35 var(--disp);
letter-spacing:.03em;color:var(--dim)}
.tile .tr.w{color:var(--text)}
.tile .sw{display:inline-block;width:3px;height:12px;border-radius:2px;margin-right:7px;
vertical-align:-1px;background:var(--faint)}

/* page */
.page{max-width:1040px;margin:0 auto;padding:0 20px 60px}
.crumb{font-size:12px;color:var(--faint);padding:14px 0 0}
.crumb a:hover{color:var(--text)}

/* game header */
.gh{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;
padding:26px 0 18px;border-bottom:1px solid var(--line)}
.team{display:flex;align-items:center;gap:14px}
.team.h{flex-direction:row-reverse;text-align:right}
.chip{width:8px;align-self:stretch;min-height:54px;border-radius:3px;background:var(--faint)}
.tn .loc{font-size:12.5px;color:var(--dim);letter-spacing:.02em}
.tn .nm{font:700 30px/1 var(--disp);letter-spacing:.02em;text-transform:uppercase}
.tn .rec{font-size:12px;color:var(--faint);margin-top:4px}
.mid{text-align:center;min-width:220px}
.score{font:700 54px/1 var(--disp);display:flex;align-items:center;justify-content:center;gap:18px}
.score .dash{color:var(--faint);font-size:30px}
.score .l{color:var(--dim)}
.status{font:600 12px/1 var(--body);letter-spacing:.06em;text-transform:uppercase;
color:var(--dim);margin-top:10px}
.status.live{color:var(--live)}
.kick{font:700 26px/1 var(--disp);letter-spacing:.03em}
.meta{text-align:center;color:var(--dim);font-size:12.5px;padding:12px 0 0}
.meta span+span:before{content:"\\00b7";margin:0 8px;color:var(--faint)}

/* section nav */
.sub{display:flex;gap:4px;margin:16px 0 4px;border-bottom:1px solid var(--line);
position:sticky;top:52px;background:var(--bg);z-index:10;overflow-x:auto}
.sub a{font:600 13px/1 var(--body);color:var(--dim);padding:11px 12px;
border-bottom:2px solid transparent;white-space:nowrap}
.sub a:hover{color:var(--text)}

/* layout */
.grid{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:18px;margin-top:18px}
@media(max-width:880px){.grid{grid-template-columns:1fr}.gh{grid-template-columns:1fr;text-align:center}
.team,.team.h{justify-content:center;flex-direction:row;text-align:left}.mid{order:-1}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;margin-bottom:18px}
.card>h3{font:700 15px/1 var(--disp);letter-spacing:.08em;text-transform:uppercase;
margin:0;padding:13px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center}
.card>h3 .x{margin-left:auto;font:500 11px/1 var(--body);letter-spacing:.02em;
text-transform:none;color:var(--faint)}
.pad{padding:14px 16px}
.empty{color:var(--faint);font-size:13px;padding:16px}

table{width:100%;border-collapse:collapse}
th,td{padding:8px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font:600 11px/1 var(--body);color:var(--faint);letter-spacing:.05em;
text-transform:uppercase;border-bottom:1px solid var(--line)}
tbody td{border-bottom:1px solid var(--row)}
tbody tr:last-child td{border-bottom:none}
td.b,th.b{font-weight:700;color:var(--text)}
.ls td:first-child{font:700 15px/1 var(--disp);letter-spacing:.04em}
.ls td.t{font:700 17px/1 var(--disp)}
.scroll{overflow-x:auto}

/* team stats comparison */
.cmp{padding:6px 16px 12px}
.cmp .row{display:grid;grid-template-columns:70px 1fr 70px;align-items:center;gap:10px;padding:7px 0;
border-bottom:1px solid var(--row)}
.cmp .row:last-child{border-bottom:none}
.cmp .v{font-weight:600}
.cmp .v.r{text-align:right}
.cmp .lab{text-align:center;font-size:12px;color:var(--dim)}
.cmp .bars{display:flex;gap:3px;margin-top:5px;height:4px}
.cmp .bars i{display:block;height:4px;border-radius:2px;background:var(--faint)}

/* box score */
.box h4{font:700 13px/1 var(--disp);letter-spacing:.08em;text-transform:uppercase;
color:var(--dim);margin:0;padding:12px 16px 4px}
.box td:first-child{font-weight:500}
.box td .pos{color:var(--faint);font-size:11px;margin-left:6px}

/* plays */
.q{font:700 12px/1 var(--disp);letter-spacing:.1em;color:var(--faint);padding:12px 16px 4px}
.play{display:grid;grid-template-columns:48px 1fr auto;gap:12px;padding:10px 16px;
border-top:1px solid var(--row);align-items:start}
.play .tm{font:700 14px/1.2 var(--disp);letter-spacing:.04em}
.play .tx{color:var(--dim);font-size:13px}
.play .tx b{color:var(--text);font-weight:600}
.play .sc{font:700 16px/1 var(--disp)}

/* leaders */
.ld{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--row)}
.ld:first-of-type{border-top:none}
.ld>div{padding:11px 16px}
.ld>div+div{border-left:1px solid var(--row)}
.ld .k{font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.05em}
.ld .p{font-weight:600;margin-top:3px}
.ld .s{color:var(--dim);font-size:12.5px}

/* model */
.mrow{display:flex;justify-content:space-between;padding:9px 16px;border-top:1px solid var(--row)}
.mrow:first-child{border-top:none}
.mrow .k{color:var(--dim)}
.mrow .v{font-weight:600}
.big{font:700 26px/1 var(--disp)}
.pill{display:inline-block;font:600 10.5px/1 var(--body);letter-spacing:.05em;text-transform:uppercase;
padding:4px 7px;border-radius:4px;background:var(--accent-tint);color:var(--accent)}
.pill.no{background:var(--bad-tint);color:var(--red)}
.lock{padding:14px 16px;color:var(--dim);font-size:13px;border-top:1px solid var(--row)}
.lock a{color:var(--accent);font-weight:600}
.wp svg{display:block;width:100%;height:auto}
.inj{display:flex;justify-content:space-between;padding:8px 16px;border-top:1px solid var(--row);font-size:13px}
.inj .s{color:var(--dim)}
.inj .s.out{color:var(--live)}
.card .card{border:none;background:none;margin:0 0 14px;padding:0}
.card .card table th:first-child{color:var(--text);font:700 14px/1 var(--disp);letter-spacing:.06em}
footer{color:var(--faint);font-size:12px;border-top:1px solid var(--line);padding-top:16px;margin-top:10px}
footer a{color:var(--dim)}
"""

JS = r"""
const API = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl';
const REN = {LA:'LAR', WAS:'WSH'};
const BACK = Object.fromEntries(Object.entries(REN).map(([a,b]) => [b,a]));
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const $ = id => document.getElementById(id);
const get = async u => { const r = await fetch(u); if (!r.ok) throw new Error(r.status); return r.json(); };
let timer = null;

function comp(ev){ return ev.competitions[0]; }
function side(ev, ha){ return comp(ev).competitors.find(c => c.homeAway === ha); }
function ourAbbr(a){ return BACK[a] || a; }
function hex(c){ return c ? '#' + c.replace('#','') : null; }
function near(a, b){
  if (!a || !b) return false;
  const p = h => [0,2,4].map(i => parseInt(h.replace('#','').slice(i,i+2), 16));
  const [x, y] = [p(a), p(b)];
  return Math.hypot(x[0]-y[0], x[1]-y[1], x[2]-y[2]) < 70;
}
function statusText(ev){
  const t = ev.status.type;
  if (t.state === 'pre') {
    const d = new Date(ev.date);
    return d.toLocaleDateString('en-US', {weekday:'short'}) + ' ' +
           d.toLocaleTimeString('en-US', {hour:'numeric', minute:'2-digit'});
  }
  return t.shortDetail;
}

function renderStrip(events){
  const strip = $('strip');
  if (!strip) return;
  const byKey = {};
  (CFG.games || []).forEach(g => byKey[g.away + '@' + g.home] = g);
  strip.innerHTML = events.map(ev => {
    const a = side(ev, 'away'), h = side(ev, 'home');
    const key = ourAbbr(a.team.abbreviation) + '@' + ourAbbr(h.team.abbreviation);
    const g = byKey[key];
    const st = ev.status.type.state;
    const done = st === 'post';
    const row = c => {
      const win = done ? c.winner : (st === 'in' ? false : true);
      const col = hex(c.team.color);
      return `<div class="tr ${done && !c.winner ? '' : 'w'}"><span><i class="sw" style="background:${col || ''}"></i>${esc(c.team.abbreviation)}</span><span>${st === 'pre' ? esc((c.records||[])[0]?.summary || '') : esc(c.score)}</span></div>`;
    };
    const cur = key === CFG.away + '@' + CFG.home;
    return `<a class="tile${cur ? ' cur' : ''}" href="${g ? g.url : '#'}">`
      + `<div class="st${st === 'in' ? ' live' : ''}">${esc(statusText(ev))}</div>`
      + row(a) + row(h) + `</a>`;
  }).join('');
  // Centre the current game by scrolling the STRIP only. scrollIntoView
  // scrolls every ancestor too, which moved the whole page and slid the tiles
  // up under the sticky nav.
  const cur = strip.querySelector('.tile.cur');
  if (cur) strip.scrollLeft = cur.offsetLeft - (strip.clientWidth - cur.clientWidth) / 2;
}

function renderHeader(ev){
  const a = side(ev, 'away'), h = side(ev, 'home');
  let ca = hex(a.team.color), ch = hex(h.team.color);
  if (near(ca, ch)) ch = hex(h.team.alternateColor) || ch;
  $('chip-a').style.background = ca || '';
  $('chip-h').style.background = ch || '';
  const rec = c => (c.records || []).map(r => r.summary).filter(Boolean)[0] || '';
  $('rec-a').textContent = rec(a);
  $('rec-h').textContent = rec(h);
  const st = ev.status.type.state;
  const mid = $('mid');
  if (st === 'pre') {
    mid.innerHTML = `<div class="kick">${esc(statusText(ev))}</div><div class="status">Kickoff</div>`;
  } else {
    const aw = +a.score, hw = +h.score;
    const lose = st === 'post' ? (aw > hw ? 'h' : (hw > aw ? 'a' : '')) : '';
    mid.innerHTML = `<div class="score"><span class="${lose === 'a' ? 'l' : ''}">${esc(a.score)}</span>`
      + `<span class="dash">-</span><span class="${lose === 'h' ? 'l' : ''}">${esc(h.score)}</span></div>`
      + `<div class="status${st === 'in' ? ' live' : ''}">${esc(ev.status.type.shortDetail)}</div>`;
  }
  const bc = (comp(ev).broadcasts || []).flatMap(b => b.names || []);
  if (bc.length) $('tv').textContent = bc.join(' / ');
}

function linescore(sum){
  const cs = sum.header?.competitions?.[0]?.competitors || [];
  const away = cs.find(c => c.homeAway === 'away'), home = cs.find(c => c.homeAway === 'home');
  if (!away?.linescores?.length) return '';
  const n = Math.max(4, away.linescores.length);
  const lab = i => i < 4 ? i + 1 : (i === 4 ? 'OT' : 'OT' + (i - 3));
  const head = Array.from({length: n}, (_, i) => `<th>${lab(i)}</th>`).join('');
  const row = c => `<tr><td>${esc(c.team.abbreviation)}</td>`
    + Array.from({length: n}, (_, i) => `<td>${esc(c.linescores[i]?.displayValue ?? '')}</td>`).join('')
    + `<td class="t">${esc(c.score)}</td></tr>`;
  return `<table class="ls"><thead><tr><th></th>${head}<th>T</th></tr></thead><tbody>${row(away)}${row(home)}</tbody></table>`;
}

function winProb(sum, ev){
  const wp = sum.winprobability || [];
  if (wp.length < 2) return '';
  const W = 640, H = 170, P = 10;
  const pts = wp.map((p, i) => [P + i * (W - 2 * P) / (wp.length - 1), P + (1 - p.homeWinPercentage) * (H - 2 * P)]);
  const line = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
  const a = side(ev, 'away'), h = side(ev, 'home');
  let ca = hex(a.team.color) || '#6b8790', ch = hex(h.team.color) || '#007a82';
  if (near(ca, ch)) ch = hex(h.team.alternateColor) || ch;
  const last = wp[wp.length - 1].homeWinPercentage;
  const lead = last >= 0.5 ? h : a, lp = last >= 0.5 ? last : 1 - last;
  return `<div class="pad" style="padding-bottom:4px;display:flex;justify-content:space-between;font-size:12px;color:var(--dim)">
      <span><b style="color:var(--text)">${esc(lead.team.abbreviation)}</b> ${(lp * 100).toFixed(1)}%</span><span>ESPN win probability</span></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Win probability through the game">
      <line x1="${P}" x2="${W-P}" y1="${H/2}" y2="${H/2}" style="stroke:var(--chart-mid)" stroke-dasharray="4 4"/>
      <text x="${P+2}" y="${P+11}" style="fill:var(--dim)" font-size="11" font-weight="700">${esc(h.team.abbreviation)}</text>
      <text x="${P+2}" y="${H-P-3}" style="fill:var(--dim)" font-size="11" font-weight="700">${esc(a.team.abbreviation)}</text>
      <path d="${line}" fill="none" style="stroke:var(--chart-line)" stroke-width="2" stroke-linejoin="round"/>
    </svg>`;
}

function leaders(sum, ev){
  // ESPN lists leaders home-first; the header reads away-left, home-right,
  // so match by team rather than trusting the order.
  const raw = sum.leaders || [];
  if (raw.length < 2) return '';
  const ab = x => x.team?.abbreviation;
  const A = side(ev, 'away').team.abbreviation, H = side(ev, 'home').team.abbreviation;
  const L = [raw.find(x => ab(x) === A) || raw[0], raw.find(x => ab(x) === H) || raw[1]];
  const want = ['passingYards', 'rushingYards', 'receivingYards'];
  const lab = {passingYards:'Passing', rushingYards:'Rushing', receivingYards:'Receiving'};
  const cell = (t, k) => {
    const cat = (t.leaders || []).find(x => x.name === k);
    const top = cat?.leaders?.[0];
    if (!top) return `<div><div class="k">${lab[k]}</div><div class="s">-</div></div>`;
    return `<div><div class="k">${esc(t.team?.abbreviation)} ${lab[k]}</div>`
      + `<div class="p">${esc(top.athlete?.shortName || top.athlete?.displayName)}</div>`
      + `<div class="s">${esc(top.displayValue)}</div></div>`;
  };
  return want.map(k => `<div class="ld">${cell(L[0], k)}${cell(L[1], k)}</div>`).join('');
}

function boxscore(sum){
  const P = sum.boxscore?.players || [];
  if (!P.length) return '';
  const groups = [['passing','Passing'], ['rushing','Rushing'], ['receiving','Receiving'],
                  ['kicking','Kicking'], ['defensive','Defense']];
  return P.map(team => {
    const blocks = groups.map(([g, title]) => {
      const s = (team.statistics || []).find(x => x.name === g);
      if (!s || !(s.athletes || []).length) return '';
      const rows = s.athletes.slice(0, g === 'defensive' ? 6 : 8).map(a =>
        `<tr><td>${esc(a.athlete?.shortName || a.athlete?.displayName)}</td>`
        + (a.stats || []).map(v => `<td>${esc(v)}</td>`).join('') + '</tr>').join('');
      return `<h4>${esc(team.team?.abbreviation)} ${title}</h4><div class="scroll"><table><thead><tr><th></th>`
        + (s.labels || []).map(l => `<th>${esc(l)}</th>`).join('') + `</tr></thead><tbody>${rows}</tbody></table></div>`;
    }).join('');
    return blocks;
  }).join('');
}

function teamStats(sum){
  const T = sum.boxscore?.teams || [];
  if (T.length < 2 || !(T[0].statistics || []).length) return '';
  const [a, h] = T;
  const hv = Object.fromEntries((h.statistics || []).map(s => [s.label, s.displayValue]));
  const keep = ['1st Downs','3rd down efficiency','4th down efficiency','Total Yards','Passing','Rushing',
                'Yards per Play','Penalties','Turnovers','Possession','Sacks-Yards Lost','Red Zone (Made-Att)'];
  const rows = (a.statistics || []).filter(s => keep.includes(s.label)).map(s => {
    const av = s.displayValue, hvv = hv[s.label] ?? '';
    const n = v => { const m = String(v).match(/^-?[\d.]+$/); return m ? parseFloat(v) : null; };
    const x = n(av), y = n(hvv);
    let bars = '';
    if (x !== null && y !== null && x + y > 0) {
      const lowerBetter = /Turnovers|Penalties|Sacks/.test(s.label);
      const aw = lowerBetter ? x < y : x > y, hw = lowerBetter ? y < x : y > x;
      bars = `<div class="bars"><i style="flex:${x};${aw ? 'background:var(--text)' : ''}"></i>`
           + `<i style="flex:${y};${hw ? 'background:var(--text)' : ''}"></i></div>`;
    }
    return `<div class="row"><div class="v">${esc(av)}</div><div class="lab">${esc(s.label)}${bars}</div><div class="v r">${esc(hvv)}</div></div>`;
  }).join('');
  return `<div class="row" style="border:none;padding-bottom:2px"><div class="v">${esc(a.team?.abbreviation)}</div><div></div><div class="v r">${esc(h.team?.abbreviation)}</div></div>` + rows;
}

function scoring(sum){
  const S = sum.scoringPlays || [];
  if (!S.length) return '';
  let q = null, out = '';
  S.forEach(p => {
    const pq = p.period?.number;
    if (pq !== q) { q = pq; out += `<div class="q">${pq <= 4 ? 'QUARTER ' + pq : 'OVERTIME'}</div>`; }
    out += `<div class="play"><div class="tm">${esc(p.team?.abbreviation)}</div>`
      + `<div class="tx"><b>${esc(p.scoringType?.displayName || p.type?.text || '')}</b> ${esc(p.clock?.displayValue || '')}<br>${esc(p.text)}</div>`
      + `<div class="sc">${esc(p.awayScore)}-${esc(p.homeScore)}</div></div>`;
  });
  return out;
}

function injuries(sum, ev){
  const A = side(ev, 'away').team.abbreviation;
  const I = [...(sum.injuries || [])].sort((x, y) => (y.team?.abbreviation === A) - (x.team?.abbreviation === A));
  const rows = I.flatMap(t => (t.injuries || []).map(x => ({t: t.team?.abbreviation, x})));
  if (!rows.length) return '';
  return rows.slice(0, 24).map(({t, x}) => {
    const s = x.status || x.details?.fantasyStatus?.description || '';
    return `<div class="inj"><span><b>${esc(t)}</b> ${esc(x.athlete?.displayName)} <span class="s">${esc(x.athlete?.position?.abbreviation || '')}</span></span>`
      + `<span class="s ${/out/i.test(s) ? 'out' : ''}">${esc(s)}</span></div>`;
  }).join('');
}

function fill(id, html, keepIfEmpty){
  const el = $(id);
  if (!el) return;
  const card = el.closest('.card');
  if (html) { el.innerHTML = html; if (card) card.hidden = false; }
  else if (!keepIfEmpty && card) card.hidden = true;
}

async function load(){
  let ev;
  try {
    const sb = await get(`${API}/scoreboard?seasontype=2&week=${CFG.week}&dates=${CFG.season}`);
    const events = sb.events || [];
    renderStrip(events);
    const A = REN[CFG.away] || CFG.away, H = REN[CFG.home] || CFG.home;
    ev = events.find(x => {
      const c = comp(x).competitors.map(k => k.team.abbreviation);
      return c.includes(A) && c.includes(H);
    });
  } catch (err) { return; }
  if (!ev) return;
  renderHeader(ev);
  const st = ev.status.type.state;
  try {
    const sum = await get(`${API}/summary?event=${ev.id}`);
    fill('ls', linescore(sum));
    fill('wp', winProb(sum, ev));
    fill('leaders', leaders(sum, ev));
    fill('box', boxscore(sum));
    fill('ts', teamStats(sum));
    fill('plays', scoring(sum));
    fill('inj', st === 'pre' ? injuries(sum, ev) : '');
    // Before kickoff ESPN's "leaders" and team stats are SEASON figures, not
    // this game's, so the headings say so.
    const pre = st === 'pre';
    const title = (id, t) => { const h = $(id)?.closest('.card')?.querySelector('h3'); if (h) h.firstChild.textContent = t; };
    title('leaders', pre ? 'Season leaders' : 'Game leaders');
    title('ts', pre ? 'Season averages' : 'Team stats');
    // Our own season table is the no-JavaScript fallback for search engines;
    // once ESPN's comparison has loaded it would only repeat it.
    if (!$('ts').closest('.card').hidden && $('season-static')) $('season-static').hidden = true;
  } catch (err) {}
  // Section links only for sections that actually have something in them.
  document.querySelectorAll('.sub a').forEach(a => {
    const t = document.querySelector(a.getAttribute('href'));
    a.hidden = !!(t && t.closest('.card') && t.closest('.card').hidden);
  });
  clearTimeout(timer);
  if (st === 'in') timer = setTimeout(load, 30000);
}
load();
"""


def _esc_json(obj):
    """JSON for embedding inside a <script> block: "</" would otherwise let a
    team or stadium name close the tag early."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def head(title, desc, canonical, ld):
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>%s</title><meta name="description" content="%s">'
        '<link rel="canonical" href="%s">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="theme-color" content="#005778">'
        '<meta property="og:type" content="article"><meta property="og:site_name" content="Phins Up">'
        '<meta property="og:title" content="%s"><meta property="og:description" content="%s">'
        '<meta property="og:url" content="%s"><meta property="og:image" content="%s/og.png">'
        '<meta name="twitter:card" content="summary_large_image">'
        % (e(title), e(desc), e(canonical), e(title), e(desc), e(canonical), SITE)
        + site_theme.HEAD_SCRIPT + FONTS
        + "<style>" + site_theme.TOKENS + site_theme.NAV_CSS + CSS + "</style>"
        + '<script type="application/ld+json">' + _esc_json(ld) + "</script></head><body>")


def nav(active="nfl"):
    return site_theme.nav(SITE, active)


def strip_static(week_games, cur_slug):
    """Server-rendered strip so the page has its matchups even without
    JavaScript; the browser replaces it with live scores."""
    out = []
    for w in week_games:
        cur = w["slug"] == cur_slug
        top = w.get("final") or ("%s %s" % ((w.get("weekday") or "")[:3], w.get("gametime") or "")).strip()
        out.append('<a class="tile%s" href="%s"><div class="st">%s</div>'
                   '<div class="tr w"><span><i class="sw"></i>%s</span><span>%s</span></div>'
                   '<div class="tr w"><span><i class="sw"></i>%s</span><span>%s</span></div></a>'
                   % (" cur" if cur else "", e(w["url"]), e(top), e(w["away"]), e(w.get("as", "")),
                      e(w["home"]), e(w.get("hs", ""))))
    return '<div class="strip" id="strip">%s</div>' % "".join(out)


def model_card(g, season, week, result):
    away = g.get("awayName") or g["awayAbbr"]
    home = g.get("homeName") or g["homeAbbr"]
    mh, eh = g.get("market_home_prob"), g.get("elo_home_prob")
    ma = None if mh is None else 1 - mh
    ea = None if eh is None else 1 - eh
    rows = []
    if g.get("primetime") and eh is not None:
        for nm, ab, ep, mp in ((away, g["awayAbbr"], ea, ma), (home, g["homeAbbr"], eh, mh)):
            ed = None if (ep is None or mp is None) else ep - mp
            rows.append('<div class="mrow"><span class="k">%s</span><span><span class="big">%s</span>'
                        '<span class="x" style="color:var(--dim);font-size:12px;margin-left:10px">'
                        'market %s%s</span></span></div>'
                        % (e(ab), pct(ep), pct(mp),
                           "" if ed is None else " &middot; %s%.1f" % ("+" if ed >= 0 else "", ed * 100)))
        body = "".join(rows)
        body += ('<div class="lock">Model win probability is free for the %s game. '
                 'Every other game is for members.</div>' % e(g["primetime"]))
    else:
        for ab, mp in ((g["awayAbbr"], ma), (g["homeAbbr"], mh)):
            rows.append('<div class="mrow"><span class="k">%s market</span><span class="v">%s</span></div>'
                        % (e(ab), pct(mp)))
        body = "".join(rows)
        body += ('<div class="lock">Our model\'s win probability for this game is for members. '
                 '<a href="%s/">Join</a> or see the <a href="%s/nfl/%s/week-%d/">free primetime games</a>.</div>'
                 % (SITE, SITE, season, week))

    a = (result or {}).get("actual")
    if (result or {}).get("graded") and a:
        ok = a.get("model_correct")
        body += ('<div class="mrow"><span class="k">Model picked</span><span class="v">%s '
                 '<span class="pill%s">%s</span></span></div>'
                 % (e(a.get("model_pick")), "" if ok else " no", "Right" if ok else "Wrong"))
        if a.get("market_pick"):
            mok = a.get("market_correct")
            body += ('<div class="mrow"><span class="k">Market picked</span><span class="v">%s '
                     '<span class="pill%s">%s</span></span></div>'
                     % (e(a["market_pick"]), "" if mok else " no", "Right" if mok else "Wrong"))
    return '<div class="card" id="model"><h3>Our model</h3>%s</div>' % body


def line_card(g):
    rows = []
    if g.get("spread_line") is not None:
        s = g["spread_line"]
        fav = g["homeAbbr"] if s > 0 else g["awayAbbr"]
        rows.append(("Spread", "%s -%s" % (fav, abs(s)) if s else "Pick"))
    if g.get("total_line") is not None:
        rows.append(("Total", "O/U %s" % g["total_line"]))
    if g.get("mlAway") is not None:
        rows.append(("%s moneyline" % g["awayAbbr"], odds(g.get("mlAway"))))
    if g.get("mlHome") is not None:
        rows.append(("%s moneyline" % g["homeAbbr"], odds(g.get("mlHome"))))
    if not rows:
        return ""
    return ('<div class="card"><h3>Line <span class="x">opening</span></h3>%s</div>'
            % "".join('<div class="mrow"><span class="k">%s</span><span class="v">%s</span></div>'
                      % (e(k), e(v)) for k, v in rows))


def page(g, season, week, result, week_games, slug, canonical, title, desc, ld):
    away = g.get("awayName") or g["awayAbbr"]
    home = g.get("homeName") or g["homeAbbr"]
    aloc = away[: -len(nickname(away, g["awayAbbr"]))].strip() or g["awayAbbr"]
    hloc = home[: -len(nickname(home, g["homeAbbr"]))].strip() or g["homeAbbr"]
    an, hn = nickname(away, g["awayAbbr"]), nickname(home, g["homeAbbr"])

    a = (result or {}).get("actual") or {}
    graded = (result or {}).get("graded") and a.get("home_score") is not None
    if graded:
        asc, hsc = a.get("away_score"), a.get("home_score")
        lose = "h" if asc > hsc else ("a" if hsc > asc else "")
        mid = ('<div class="score"><span class="%s">%s</span><span class="dash">-</span>'
               '<span class="%s">%s</span></div><div class="status">Final</div>'
               % ("l" if lose == "a" else "", e(asc), "l" if lose == "h" else "", e(hsc)))
    else:
        mid = ('<div class="kick">%s %s</div><div class="status">Kickoff ET</div>'
               % (e((g.get("weekday") or "")[:3]), e(g.get("gametime") or "")))

    meta = ["<span>%s</span>" % e(pretty_date(g.get("gameday")))]
    if g.get("stadium"):
        meta.append("<span>%s</span>" % e(g["stadium"]))
    meta.append('<span id="tv"></span>')

    cfg = {"season": season, "week": week, "away": g["awayAbbr"], "home": g["homeAbbr"],
           "games": [{"away": w["away"], "home": w["home"], "url": w["url"]} for w in week_games]}

    stats = (team_stats_table(g.get("awayTeamStats"), away)
             + team_stats_table(g.get("homeTeamStats"), home))

    return "".join([
        head(title, desc, canonical, ld),
        nav("nfl"),
        strip_static(week_games, slug),
        '<main class="page">',
        '<div class="crumb"><a href="%s/nfl/%s/">NFL %s</a> / <a href="%s/nfl/%s/week-%d/">Week %d</a></div>'
        % (SITE, season, season, SITE, season, week, week),
        '<h1 style="position:absolute;left:-9999px">%s at %s, NFL Week %d</h1>' % (e(an), e(hn), week),
        '<section class="gh">',
        '<div class="team"><div class="chip" id="chip-a"></div><div class="tn">'
        '<div class="loc">%s</div><div class="nm">%s</div><div class="rec" id="rec-a"></div></div></div>'
        % (e(aloc), e(an)),
        '<div class="mid" id="mid">%s</div>' % mid,
        '<div class="team h"><div class="chip" id="chip-h"></div><div class="tn">'
        '<div class="loc">%s</div><div class="nm">%s</div><div class="rec" id="rec-h"></div></div></div>'
        % (e(hloc), e(hn)),
        '</section>',
        '<div class="meta">%s</div>' % "".join(meta),
        '<nav class="sub"><a href="#summary">Summary</a><a href="#boxscore">Box score</a>'
        '<a href="#teamstats">Team stats</a><a href="#scoring">Scoring</a><a href="#model">Our model</a></nav>',
        '<div class="grid"><div>',
        '<div class="card" id="summary" hidden><h3>Score by quarter</h3><div class="scroll" id="ls"></div></div>',
        '<div class="card wp" hidden><h3>Win probability</h3><div id="wp"></div></div>',
        '<div class="card" hidden><h3>Game leaders</h3><div id="leaders"></div></div>',
        '<div class="card box" id="boxscore" hidden><h3>Box score</h3><div id="box"></div></div>',
        '<div class="card" id="teamstats" hidden><h3>Team stats</h3><div class="cmp" id="ts"></div></div>',
        '<div class="card" id="scoring" hidden><h3>Scoring plays</h3><div id="plays"></div></div>',
        '<div class="card" hidden><h3>Injury report</h3><div id="inj"></div></div>',
        ('<div class="card" id="season-static"><h3>Season stats</h3><div class="pad">%s</div></div>' % stats) if stats else "",
        '</div><aside>',
        model_card(g, season, week, result),
        line_card(g),
        '</aside></div>',
        '<footer>Statistical projections for information and entertainment, not betting advice. '
        'Live stats from ESPN. Not affiliated with the NFL or any sportsbook. '
        '&middot; <a href="%s/track-record">Track record</a> &middot; '
        '<a href="https://x.com/PhinsUpDotNet" rel="noopener">@PhinsUpDotNet</a></footer>' % SITE,
        '</main>',
        site_theme.TOGGLE_JS,
        '<script>const CFG=%s;</script><script>%s</script>' % (_esc_json(cfg), JS),
        '</body></html>',
    ])

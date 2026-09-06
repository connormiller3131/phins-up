/**
 * Builds the FREE-ACCOUNT half of the gated payload.
 *
 * The account tier is "the model's picks" -- suggested parlay, TD/HR Special,
 * same-game parlays, weekly picks -- while player-prop tables are the paid
 * tier. Awkwardly, every one of those cards is computed IN THE BROWSER from
 * props, so the picks cannot be served without the props they came from.
 *
 * The obvious fix is to precompute each card server-side, but that means a
 * second implementation of the selection logic that has to agree with the
 * browser's forever. Instead this ships a REDUCED PROPS payload: exactly the
 * props the cards select, and nothing else. The page's own unmodified card
 * code then produces identical cards from it, so there is still only one
 * implementation of selection anywhere. The props tables are hidden for
 * non-subscribers, which is what actually separates the tiers.
 *
 * This rests on an invariant: picking the best item from a pool containing
 * the winners gives the same answer as picking from the full pool, because
 * everything omitted ranked below something kept. That holds for top-N
 * ranking but is not free -- verify_account_split.mjs checks the cards come
 * out identical rather than trusting it.
 *
 * Run after build_static_site.py, which writes the paid payload this reads.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const TEMPLATE = path.join(ROOT, 'pipeline', 'nfl', 'dashboard_live.html');
const NFL_DATA_PATH = path.join(ROOT, 'data', 'nfl', 'dashboard_current_week.json');
const MLB_DATA_PATH = path.join(ROOT, 'data', 'mlb', 'dashboard_current_slate.json');
const PAID_PATH = path.join(ROOT, 'data', 'gated_payload.json');
const OUT_PATH = path.join(ROOT, 'data', 'gated_payload_account.json');

const tpl = fs.readFileSync(TEMPLATE, 'utf8');

/** Slice one top-level declaration out of the page by name. */
function extract(decl) {
  const i = tpl.indexOf(decl);
  if (i < 0) throw new Error(`not found in template: ${decl}`);
  if (decl.startsWith('function') || decl.startsWith('async function')) {
    let d = 0, started = false;
    for (let j = i; j < tpl.length; j++) {
      if (tpl[j] === '{') { d++; started = true; }
      else if (tpl[j] === '}') { d--; if (started && d === 0) return tpl.slice(i, j + 1); }
    }
  } else {
    let d = 0;
    for (let j = i; j < tpl.length; j++) {
      const c = tpl[j];
      if (c === '{' || c === '[' || c === '(') d++;
      else if (c === '}' || c === ']' || c === ')') d--;
      else if (c === ';' && d === 0) return tpl.slice(i, j + 1);
    }
  }
  throw new Error(`unbalanced: ${decl}`);
}

// Everything the pick cards need, pulled from the page itself so there is
// exactly one definition of how a pick is chosen.
const NEEDED = [
  // Formatting helpers the pick builders reach for. They only shape display
  // strings, but nflWeeklyPicksGroups calls pct() while building its rows, so
  // it cannot run without them.
  'function pct', 'function fmtOdds',
  'const RELIABLE_TRAILING_N', 'function isReliablePick',
  'const LADDER_STEP_DOWN_MARKETS', 'function pickLine', 'function bestPropInMarket',
  'const TD_MIN_OPP_PER_GM', 'function isFeaturedTdLeg',
  'const NFL_PARLAY_MONEYLINES', 'const NFL_PARLAY_PROPS', 'const NFL_PARLAY_PROP_MARKETS',
  'function collectNflParlayLegs', 'function collectNflTdSpecialLegs',
  'function collectNflSameGameParlayLegs',
  'const NFL_PICK_MARKETS', 'function nflWeeklyPicksGroups',
  'function noVigProb', 'function buildRealisticParlay', 'function collectMlbParlayLegs',
  'function collectHrSpecialLegs', 'function collectMlbSameGameParlayLegs',
  'function mlbPrevDateStr', 'function mlbHrSpecialPicksForDate',
];

export function loadSelectors(DATA, MLB_DATA) {
  const src = NEEDED.map(extract).join('\n\n');
  const names = NEEDED.map((d) => d.replace(/^(?:async )?function |^const /, '').trim());
  return new Function('DATA', 'MLB_DATA', `${src}\nreturn {${names.join(',')}};`)(DATA, MLB_DATA);
}

/**
 * Every prop the account-tier cards reference, as a set of player ids per
 * game. Runs the real selectors over the FULL data, then records who they
 * chose. Flin Flon is deliberately excluded: it draws at random on every page
 * load, so no fixed subset can reproduce it, and it is a novelty rather than
 * part of the tier.
 */
// Keyed per PROP, not per player: "<player_id>|<market>". Keeping a player
// wholesale would hand over every market they appear in -- a third of the
// board on real data -- when the cards only ever reference one line each.
const propKey = (playerId, market) => `${playerId}|${market}`;

export function collectKeptIds(DATA, MLB_DATA, sel) {
  const nfl = new Map();   // "week|away|home" -> Set("<player_id>|<market>")
  const mlb = new Map();   // gamePk           -> Set("<player_id>|<market>")

  const keep = (map, key, playerId, market) => {
    if (playerId == null || !market) return;
    if (!map.has(key)) map.set(key, new Set());
    map.get(key).add(propKey(playerId, market));
  };
  const nflKey = (week, g) => `${week}|${g.awayAbbr}|${g.homeAbbr}`;
  // A leg does not record which game it came from, so find the game holding
  // that exact prop. Matching on market as well as player matters: a running
  // back appears in Rushing Yds, Carries and Anytime TD, and only the one the
  // card actually used should travel.
  const scatter = (map, keyOf, games, legs) => {
    for (const leg of legs || []) {
      if (leg.player_id == null || !leg.market) continue;
      const g = games.find((x) => (x.props || [])
        .some((p) => p.player_id === leg.player_id && p.market === leg.market));
      if (g) keep(map, keyOf(g), leg.player_id, leg.market);
    }
  };

  for (const [week, wk] of Object.entries(DATA.weeks || {})) {
    const games = wk.games || [];
    const keyOf = (g) => nflKey(week, g);
    scatter(nfl, keyOf, games, sel.collectNflParlayLegs(games));
    scatter(nfl, keyOf, games, sel.collectNflTdSpecialLegs(games));
    for (const g of games) scatter(nfl, keyOf, [g], sel.collectNflSameGameParlayLegs(g));

    // The parlay picks a winner in THREE markets and then keeps only the best
    // two, so one winner is computed and discarded. It still has to travel:
    // without it, a lesser prop kept for some other card becomes that
    // market's winner in the smaller pool, and can outrank a real pick. This
    // was not hypothetical -- it changed four weeks' parlays before the
    // verifier caught it. Replays the same per-market loop, with the same
    // accumulating `used` set, and keeps every winner including the loser.
    const pool = games.filter((g) => !g.already_played).flatMap((g) => g.props || []);
    const used = new Set();
    for (const mk of sel.NFL_PARLAY_PROP_MARKETS) {
      const pick = sel.bestPropInMarket(pool, mk, used);
      if (!pick) continue;
      used.add(pick.player_id);
      scatter(nfl, keyOf, games, [pick]);
    }
  }

  // Weekly picks is a leaderboard, and its rows are display objects with no
  // player_id or market on them -- so its selection is replayed here against
  // the props directly, using the page's own NFL_PICK_MARKETS and the same
  // sort. Reading the rows instead would silently keep nothing, which is
  // exactly what the first version of this did.
  const cw = DATA.weeks?.[String(DATA.current_week)];
  if (cw) {
    const games = cw.games || [];
    const all = games.flatMap((g) => g.props || []);
    for (const cfg of sel.NFL_PICK_MARKETS) {
      const rows = all
        .filter((p) => p.market === cfg.market &&
          (cfg.binary ? p.model_prob != null : p.projected != null))
        .sort((a, b) => (cfg.binary ? b.model_prob - a.model_prob : b.projected - a.projected))
        .slice(0, 3);
      for (const p of rows) {
        const g = games.find((x) => (x.props || []).includes(p));
        if (g) keep(nfl, nflKey(String(DATA.current_week), g), p.player_id, p.market);
      }
    }
  }

  for (const [date, day] of Object.entries(MLB_DATA.days || {})) {
    const games = day.games || [];
    const keyOf = (g) => String(g.gamePk);
    scatter(mlb, keyOf, games, sel.collectMlbParlayLegs(games));
    scatter(mlb, keyOf, games, sel.mlbHrSpecialPicksForDate(date));
    for (const g of games) scatter(mlb, keyOf, [g], sel.collectMlbSameGameParlayLegs(g));

    // Same discard problem on the MLB side: buildRealisticParlay reserves
    // Total Bases, then draws fillers from Anytime HR / Hits / RBI and caps
    // the card, so a computed winner can be dropped. The __lineup_known flag
    // is reproduced because bestPropInMarket reads it to decide whether a
    // bench player is eligible, and getting that wrong here would keep the
    // wrong prop.
    const pool = games.flatMap((g) => {
      const lineupKnown = (g.props || []).some((p) => p.section === 'Batting' && p.confirmed_starter === true);
      return (g.props || []).map((p) => (p.section === 'Batting' ? { ...p, __lineup_known: lineupKnown } : p));
    });
    const usedMlb = new Set();
    for (const mk of ['Total Bases', 'Anytime HR', 'Hits', 'RBI']) {
      const pick = sel.bestPropInMarket(pool, mk, usedMlb);
      if (!pick) continue;
      usedMlb.add(pick.player_id);
      scatter(mlb, keyOf, games, [pick]);
    }
  }
  return { nfl, mlb };
}

function reduce(paidSection, keptBySection, extraKeys) {
  const out = {};
  let kept = 0, dropped = 0;
  for (const [key, entry] of Object.entries(paidSection || {})) {
    const ids = keptBySection.get(key);
    const props = (entry.props || [])
      .filter((p) => ids && ids.has(propKey(p.player_id, p.market)));
    dropped += (entry.props || []).length - props.length;
    kept += props.length;
    if (!props.length && !extraKeys.some((k) => entry[k] != null)) continue;
    out[key] = { ...entry, props };
  }
  return { out, kept, dropped };
}

function main() {
  const DATA = JSON.parse(fs.readFileSync(NFL_DATA_PATH, 'utf8'));
  const MLB_DATA = fs.existsSync(MLB_DATA_PATH)
    ? JSON.parse(fs.readFileSync(MLB_DATA_PATH, 'utf8')) : { days: {} };
  const paid = JSON.parse(fs.readFileSync(PAID_PATH, 'utf8'));

  const sel = loadSelectors(DATA, MLB_DATA);
  const { nfl, mlb } = collectKeptIds(DATA, MLB_DATA, sel);

  // hr_combo is the MLB "either/or homer" line shown beside the HR Special,
  // so it belongs to the account tier and is carried through.
  const n = reduce(paid.nfl, nfl, []);
  const m = reduce(paid.mlb, mlb, ['hr_combo']);

  const account = { build: paid.build, nfl: n.out, mlb: m.out };
  fs.writeFileSync(OUT_PATH, JSON.stringify(account));

  const size = (p) => (fs.statSync(p).size / 1e6).toFixed(2);
  console.log(`account payload: ${size(OUT_PATH)} MB vs paid ${size(PAID_PATH)} MB`);
  console.log(`  NFL games ${Object.keys(n.out).length}, props kept ${n.kept}, withheld ${n.dropped}`);
  console.log(`  MLB games ${Object.keys(m.out).length}, props kept ${m.kept}, withheld ${m.dropped}`);
}

if (import.meta.url === pathToFileURLSafe(process.argv[1])) main();

function pathToFileURLSafe(p) {
  try { return new URL(`file://${path.resolve(p).replace(/\\/g, '/')}`).href; }
  catch { return ''; }
}

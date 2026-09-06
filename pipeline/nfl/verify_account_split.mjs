/**
 * The account payload rests on one invariant: selecting the best pick from a
 * pool that contains the winners gives the same answer as selecting from the
 * full pool. That is true for top-N ranking, but "should be true" is not the
 * same as true -- a `used` set, a different-teams rule or a tie-break could
 * all behave differently on a smaller pool.
 *
 * So this rebuilds the games exactly as a free account's browser will see
 * them (props replaced by the reduced set) and re-runs every pick card, then
 * compares against the same cards built from the full data. Any difference
 * means a free account would be shown a DIFFERENT pick from the one the model
 * actually chose, which is worse than showing nothing.
 *
 * Run after build_account_payload.mjs. Exits non-zero on any mismatch so it
 * can fail a CI run rather than shipping a quietly wrong card.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadSelectors } from './build_account_payload.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));

const FULL_NFL = read('data/nfl/dashboard_current_week.json');
const FULL_MLB = fs.existsSync(path.join(ROOT, 'data/mlb/dashboard_current_slate.json'))
  ? read('data/mlb/dashboard_current_slate.json') : { days: {} };
const account = read('data/gated_payload_account.json');

/** Deep copy with props swapped for whatever the account payload carries. */
function asFreeAccountSees(nflData, mlbData, acct) {
  const nfl = JSON.parse(JSON.stringify(nflData));
  const mlb = JSON.parse(JSON.stringify(mlbData));
  for (const [week, wk] of Object.entries(nfl.weeks || {})) {
    for (const g of wk.games || []) {
      const entry = acct.nfl[`${week}|${g.awayAbbr}|${g.homeAbbr}`];
      g.props = entry ? entry.props : [];
    }
  }
  for (const day of Object.values(mlb.days || {})) {
    for (const g of day.games || []) {
      const entry = acct.mlb[String(g.gamePk)];
      g.props = entry ? entry.props : [];
      if (entry && entry.hr_combo !== undefined) g.hr_combo = entry.hr_combo;
    }
  }
  return { nfl, mlb };
}

const fingerprint = (legs) => (legs || []).map((l) =>
  `${l.market}|${l.label}|${l.sub}|${(l.prob ?? 0).toFixed(4)}|${l.edge == null ? '-' : l.edge.toFixed(4)}`
).join(' ~ ');

function allCards(DATA, MLB_DATA) {
  const sel = loadSelectors(DATA, MLB_DATA);
  const out = {};
  for (const [week, wk] of Object.entries(DATA.weeks || {})) {
    const games = wk.games || [];
    out[`nfl:${week}:parlay`] = fingerprint(sel.collectNflParlayLegs(games));
    out[`nfl:${week}:td`] = fingerprint(sel.collectNflTdSpecialLegs(games));
    games.forEach((g, i) => {
      out[`nfl:${week}:sgp:${i}`] = fingerprint(sel.collectNflSameGameParlayLegs(g));
    });
  }
  const cw = DATA.weeks?.[String(DATA.current_week)];
  if (cw) {
    out['nfl:weeklyPicks'] = JSON.stringify(sel.nflWeeklyPicksGroups(cw));
  }
  for (const [date, day] of Object.entries(MLB_DATA.days || {})) {
    const games = day.games || [];
    out[`mlb:${date}:parlay`] = fingerprint(sel.collectMlbParlayLegs(games));
    out[`mlb:${date}:hr`] = fingerprint(sel.mlbHrSpecialPicksForDate(date));
    games.forEach((g, i) => {
      out[`mlb:${date}:sgp:${i}`] = fingerprint(sel.collectMlbSameGameParlayLegs(g));
    });
  }
  return out;
}

const full = allCards(FULL_NFL, FULL_MLB);
const { nfl, mlb } = asFreeAccountSees(FULL_NFL, FULL_MLB, account);
const reduced = allCards(nfl, mlb);

const keys = [...new Set([...Object.keys(full), ...Object.keys(reduced)])];
const diffs = keys.filter((k) => full[k] !== reduced[k]);

console.log(`cards compared: ${keys.length}`);
console.log(`identical:      ${keys.length - diffs.length}`);
console.log(`different:      ${diffs.length}`);
if (diffs.length) {
  for (const k of diffs.slice(0, 12)) {
    console.log(`\n  ${k}`);
    console.log(`    full    : ${String(full[k]).slice(0, 160) || '(empty)'}`);
    console.log(`    reduced : ${String(reduced[k]).slice(0, 160) || '(empty)'}`);
  }
  if (diffs.length > 12) console.log(`\n  ...and ${diffs.length - 12} more`);
  process.exit(1);
}
console.log('\nAccount payload reproduces every pick card exactly.');

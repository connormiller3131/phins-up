"""Generate REAL projections for the next day's MLB slate:
- Elo win probability, ratings carried forward from all completed games,
  blended with real starting-pitcher/bullpen strength.
- Real DraftKings game-line odds (moneyline/run-line/total) via The Odds API.
- Player props for each team's probable starting pitcher and the full
  9-player lineup by recent plate-appearance volume (a proxy for "everyday
  player" -- we don't have a confirmed lineup source), sectioned into
  Batting and Pitching:
    Batting: Hits, Total Bases, Walks, RBI, Anytime HR
    Pitching: Strikeouts, Hits Allowed, Walks Allowed, Runs Allowed,
              Outs Recorded
  fit ONCE per stat on the full historical dataset and reused across every
  game (the target date is genuinely in the future, so there's no leakage
  to guard against the way there is in a backtest). Hits and Total Bases
  additionally show a small ladder of lines around the trailing average.
- Real DraftKings player-prop odds for a handful of games each refresh
  (per-event pulls cost credits per market, so not the whole slate) --
  Anytime HR gets a TAKE/MAYBE/PASS/RISKY grade against the real price,
  same edge-based tiers as NFL's Anytime TD.
- An either/or anytime-HR "special": P(at least one of two players homers),
  computed from our own anytime-HR model for the single best HR bet on each
  team, assuming independence between the two players.
- A GOOD VALUE tag on the game line when Model win % beats the market's
  fair % for that side.
"""
import os
import sys
import pathlib
import json
import datetime
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.elo_model import run_elo
from pipeline.mlb.team_map import full_name_to_statcast, STATCAST_TO_MLB_TEAM_ID
from pipeline.mlb.props.prop_data import build_batter_prop_table, build_pitcher_prop_table
from pipeline.mlb.props.current_state import (
    batter_current_trailing, batter_opponent_current_trailing,
    pitcher_current_trailing, pitcher_opponent_current_trailing,
)
from pipeline.mlb.props.prop_models import FEATURES, over_prob
from pipeline.mlb.pitcher_ratings import current_sp_rating, current_bullpen_rating
from pipeline.common.odds_api import get_game_odds, get_event_player_props

DATA_DIR = ROOT / "data" / "mlb"

BATTER_COUNT_STATS = {
    "hits": "Hits", "total_bases": "Total Bases", "walks": "Walks", "rbi": "RBI",
}
LADDER_STATS = {"hits", "total_bases"}
PITCHER_COUNT_STATS = {
    "strikeouts": "Pitcher Strikeouts", "hits_allowed": "Pitcher Hits Allowed",
    "walks_allowed": "Pitcher Walks Allowed", "runs_allowed": "Pitcher Runs Allowed",
    "outs_recorded": "Pitcher Outs Recorded",
}

# Only stats with a real matching DraftKings market get odds wired up.
# "runs_allowed" has no direct market (books price earned runs, which needs
# official-scorer judgment we can't derive from pitch-level data), so it
# stays model-only, same treatment as NFL's non-TD yardage props.
PROP_MARKET_KEYS = {
    "Hits": "batter_hits", "Total Bases": "batter_total_bases", "Walks": "batter_walks",
    "RBI": "batter_rbis", "Anytime HR": "batter_home_runs",
    "Pitcher Strikeouts": "pitcher_strikeouts", "Pitcher Hits Allowed": "pitcher_hits_allowed",
    "Pitcher Walks Allowed": "pitcher_walks", "Pitcher Outs Recorded": "pitcher_outs",
}


def get_slate_schedule(max_days_ahead=5):
    for offset in range(max_days_ahead):
        d = (datetime.date.today() + datetime.timedelta(days=offset)).isoformat()
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": d, "hydrate": "probablePitcher"},
            timeout=15,
        )
        resp.raise_for_status()
        dates = resp.json().get("dates", [])
        if dates and dates[0]["games"]:
            return d, dates[0]["games"]
    raise RuntimeError("No MLB games found in the next several days.")


def parse_slate(raw_games):
    out = []
    for g in raw_games:
        away, home = g["teams"]["away"], g["teams"]["home"]
        try:
            away_abbr = full_name_to_statcast(away["team"]["name"])
            home_abbr = full_name_to_statcast(home["team"]["name"])
        except KeyError as e:
            print(f"  skip game, unmapped team: {e}")
            continue
        out.append({
            "away_name": away["team"]["name"], "home_name": home["team"]["name"],
            "away_team": away_abbr, "home_team": home_abbr,
            "away_probable_pitcher": away.get("probablePitcher"),
            "home_probable_pitcher": home.get("probablePitcher"),
            "game_datetime": g.get("gameDate"),
            "game_pk": g.get("gamePk"),
        })
    return out


def elo_predictions(games_df, slate):
    with open(ROOT / "notebooks_out" / "mlb_win_prob_backtest.json") as f:
        elo_params = json.load(f)["elo_params"]

    today = pd.Timestamp.today().normalize()
    future_rows = pd.DataFrame({
        "season": [today.year] * len(slate),
        "game_date": [today] * len(slate),
        "home_team": [g["home_team"] for g in slate],
        "away_team": [g["away_team"] for g in slate],
        "margin": np.nan, "home_win": np.nan,
    })
    cols = ["season", "game_date", "home_team", "away_team", "margin", "home_win"]
    combined = pd.concat([games_df[cols], future_rows], ignore_index=True)
    preds = run_elo(combined, k=elo_params["k"], home_adv=elo_params["home_adv"], scale=elo_params["scale"],
                    season_regression=elo_params.get("season_regression", 0.65))
    return preds[-len(slate):], elo_params


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return np.log(p / (1 - p))


def blend_with_pitcher_strength(elo_preds, slate):
    """Refine the team-only Elo prediction with real starting-pitcher and
    bullpen strength, using the fitted blend from backtest_pitcher_model.py
    (validated out-of-sample: Brier 0.2486 -> 0.2480 on the 2026 holdout).
    Falls back to the pure Elo number if the backtest artifact is missing."""
    path = ROOT / "notebooks_out" / "mlb_pitcher_model_backtest.json"
    if not path.exists():
        return list(elo_preds)
    with open(path) as f:
        b = json.load(f)
    coef, intercept = b["coef"], b["intercept"]
    sp_fill, bp_fill = b["sp_fill"], b["bp_fill"]

    out = []
    for i, g in enumerate(slate):
        home_pid = g["home_probable_pitcher"]["id"] if g["home_probable_pitcher"] else None
        away_pid = g["away_probable_pitcher"]["id"] if g["away_probable_pitcher"] else None
        home_sp = current_sp_rating(home_pid) if home_pid else None
        away_sp = current_sp_rating(away_pid) if away_pid else None
        home_bp = current_bullpen_rating(g["home_team"])
        away_bp = current_bullpen_rating(g["away_team"])

        sp_diff = (home_sp - away_sp) if (home_sp is not None and away_sp is not None) else sp_fill
        bp_diff = (home_bp - away_bp) if (home_bp is not None and away_bp is not None) else bp_fill

        z = coef[0] * logit(float(elo_preds[i])) + coef[1] * sp_diff + coef[2] * bp_diff + intercept
        out.append(1.0 / (1.0 + np.exp(-z)))
    return out


def _implied_prob(american_odds):
    return (-american_odds) / (-american_odds + 100) if american_odds < 0 else 100 / (american_odds + 100)


def attach_market_odds(slate):
    try:
        odds_data = get_game_odds("baseball_mlb")
    except Exception as e:
        print(f"  odds API unavailable: {e}")
        for g in slate:
            g["market"] = None
        return

    # A team pair can appear more than once in a day (doubleheaders), so keep
    # every match per pair and pick the closest commence_time per slate game
    # rather than a plain dict overwrite (which would silently drop one game).
    by_pair = {}
    for g in odds_data:
        try:
            h, a = full_name_to_statcast(g["home_team"]), full_name_to_statcast(g["away_team"])
        except KeyError:
            continue
        by_pair.setdefault((a, h), []).append(g)

    for slate_game in slate:
        candidates = by_pair.get((slate_game["away_team"], slate_game["home_team"]), [])
        odds_game = None
        if len(candidates) == 1:
            odds_game = candidates[0]
        elif len(candidates) > 1 and slate_game.get("game_datetime"):
            target = pd.Timestamp(slate_game["game_datetime"])
            odds_game = min(candidates, key=lambda g: abs(pd.Timestamp(g["commence_time"]) - target))
        if not odds_game or not odds_game.get("bookmakers"):
            slate_game["market"] = None
            continue
        dk = odds_game["bookmakers"][0]
        markets = {m["key"]: m for m in dk["markets"]}
        ml_home = ml_away = run_line_home = total_line = None
        if "h2h" in markets:
            for o in markets["h2h"]["outcomes"]:
                if o["name"] == odds_game["home_team"]:
                    ml_home = o["price"]
                elif o["name"] == odds_game["away_team"]:
                    ml_away = o["price"]
        if "spreads" in markets:
            for o in markets["spreads"]["outcomes"]:
                if o["name"] == odds_game["home_team"]:
                    run_line_home = o["point"]
        if "totals" in markets:
            total_line = markets["totals"]["outcomes"][0].get("point")
        slate_game["market"] = {
            "mlHome": ml_home, "mlAway": ml_away,
            "run_line_home": run_line_home, "total_line": total_line,
            "event_id": odds_game.get("id"), "commence_time": odds_game.get("commence_time"),
        }
        if ml_home is not None and ml_away is not None:
            pa, ph = _implied_prob(ml_away), _implied_prob(ml_home)
            slate_game["market"]["home_fair_prob"] = round(ph / (pa + ph), 4)


def _norm_name(name):
    """Accent-insensitive lowercase match key (DK strips accents: Jose Ramirez)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c)).lower().strip()


def attach_featured_prop_odds(games_out, limit=2):
    """Fetch real DraftKings player-prop odds for the most competitive
    not-yet-started game(s) only -- each per-event props call costs credits
    per market requested, so the free tier can afford a couple of games per
    refresh, not the whole slate."""
    now = pd.Timestamp.utcnow()
    candidates = [
        g for g in games_out
        if g.get("market") and g["market"].get("event_id")
        and g["market"].get("commence_time") and pd.Timestamp(g["market"]["commence_time"]) > now
    ]
    candidates.sort(key=lambda g: abs(g["elo_home_prob"] - 0.5))

    for g in candidates[:limit]:
        try:
            data = get_event_player_props(
                "baseball_mlb", g["market"]["event_id"],
                markets=",".join(sorted(set(PROP_MARKET_KEYS.values()))))
        except Exception as e:
            print(f"  featured props fetch failed for {g['awayAbbr']} @ {g['homeAbbr']}: {e}")
            continue

        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            continue
        dk_markets = {m["key"]: m for m in bookmakers[0].get("markets", [])}

        # (market_key, normalized player name) -> {line, over, under}
        odds_lookup = {}
        for mkey, market in dk_markets.items():
            for o in market.get("outcomes", []):
                player = _norm_name(o.get("description", ""))
                entry = odds_lookup.setdefault((mkey, player), {})
                entry["line"] = o.get("point")
                if o.get("name") == "Over":
                    entry["over"] = o.get("price")
                elif o.get("name") == "Under":
                    entry["under"] = o.get("price")

        matched = 0
        for p in g["props"]:
            mkey = PROP_MARKET_KEYS.get(p["market"])
            if not mkey:
                continue
            dk = odds_lookup.get((mkey, _norm_name(p["player"])))
            if not dk or dk.get("over") is None:
                continue
            p["dk_line"] = dk.get("line")
            p["dk_over"] = dk.get("over")
            p["dk_under"] = dk.get("under")
            p["dk_implied_prob"] = round(_implied_prob(dk["over"]), 4)
            # Re-aim the model probability at DraftKings' actual line instead
            # of our trailing-average proxy (binary Anytime HR already IS
            # P(over 0.5), so only count props need recomputing).
            if p["market"] != "Anytime HR" and dk.get("line") is not None and "model_std" in p:
                p["model_over_prob"] = round(float(over_prob(p["projected"], p["model_std"], dk["line"])), 3)
                p["line"] = dk["line"]
            matched += 1

        g["featured_props"] = True
        print(f"  featured prop odds: {g['awayAbbr']} @ {g['homeAbbr']} ({matched} props matched to DK)")


def prepare_count_model(hist_df):
    model = RidgeCV(alphas=np.logspace(-1, 3, 25))
    model.fit(hist_df[FEATURES].values, hist_df["actual"].values)
    resid_std = max(float(np.std(hist_df["actual"].values - model.predict(hist_df[FEATURES].values))), 1e-6)
    return model, resid_std


def prepare_binary_model(hist_df):
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(hist_df[FEATURES].values, hist_df["actual"].values)
    return model


def count_ladder(pred_mean, resid_std, own_avg, step=1.0, n=3):
    """A small ladder of lines in steps of 1 around the player's trailing
    average -- MLB counting stats (hits, total bases) are small integers, so
    NFL's steps-of-10 doesn't translate; steps of 1 does."""
    base = round(own_avg * 2) / 2
    half = (n // 2) * step
    out = []
    for i in range(n):
        line = round(base - half + i * step, 1)
        if line <= 0:
            continue
        out.append({"line": line, "over_prob": round(float(over_prob(pred_mean, resid_std, line)), 3)})
    return out


def project_count_stat(stat_key, prep, player_id, opp_team):
    own, opp = prep["own"], prep["opp"]
    if player_id not in own.index or opp_team not in opp.index:
        return None
    own_avg = own.loc[player_id, "current_avg"]
    if pd.isna(own_avg):
        return None
    opp_avg = float(opp.loc[opp_team])
    pred_mean = float(prep["model"].predict([[float(own_avg), opp_avg]])[0])
    line = round(own_avg * 2) / 2
    p_over = float(over_prob(pred_mean, prep["std"], line))
    out = {
        "line": line, "projected": round(pred_mean, 1), "model_over_prob": round(p_over, 3),
        "model_std": round(prep["std"], 3), "player_display_name": own.loc[player_id, "player_display_name"],
    }
    if stat_key in LADDER_STATS:
        out["ladder"] = count_ladder(pred_mean, prep["std"], float(own_avg))
    return out


def get_active_roster_ids(team_abbr):
    """Real current 26-man active roster (excludes injured/optioned/DFA'd
    players) -- fixes recommending someone like an injured star whose recent
    trailing stats still look great but who literally isn't playing."""
    team_id = STATCAST_TO_MLB_TEAM_ID.get(team_abbr)
    if not team_id:
        return None
    try:
        resp = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                            params={"rosterType": "active"}, timeout=15)
        resp.raise_for_status()
        return {p["person"]["id"] for p in resp.json().get("roster", [])}
    except Exception as e:
        print(f"  roster fetch failed for {team_abbr}: {e}")
        return None  # fail open: don't filter if the roster call itself breaks


def top_batters_for_team(pa_own, team, active_ids, n=9):
    """Full lineup by recent plate-appearance volume. No confirmed daily
    lineup source (same caveat as before), so this is a proxy for "everyday
    player" -- 9 players covers a real lineup, not just the top few bats."""
    team_players = pa_own[pa_own["team"] == team]
    if active_ids is not None:
        team_players = team_players[team_players.index.isin(active_ids)]
    team_players = team_players.sort_values("current_avg", ascending=False)
    return team_players.head(n).index.tolist()


def batter_props_for_team(team, opp_team, player_ids, batter_models, hr_model, hr_own, hr_opp):
    entries = []
    hr_candidates = []

    for pid in player_ids:
        name = None
        for stat_key, market_label in BATTER_COUNT_STATS.items():
            r = project_count_stat(stat_key, batter_models[stat_key], pid, opp_team)
            if r:
                name = r["player_display_name"]
                entries.append({"section": "Batting", "player": name, "player_id": int(pid), "team": team,
                                 "market": market_label, "line": r["line"], "projected": r["projected"],
                                 "model_over_prob": r["model_over_prob"], "model_std": r["model_std"],
                                 **({"ladder": r["ladder"]} if "ladder" in r else {})})

        if pid in hr_own.index and opp_team in hr_opp.index:
            own_hr = hr_own.loc[pid, "current_avg"]
            if pd.notna(own_hr):
                name = name or hr_own.loc[pid, "player_display_name"]
                hr_prob = float(hr_model.predict_proba([[float(own_hr), float(hr_opp.loc[opp_team])]])[:, 1][0])
                entries.append({"section": "Batting", "player": name, "player_id": int(pid), "team": team,
                                 "market": "Anytime HR", "model_prob": round(hr_prob, 3)})
                hr_candidates.append((name, hr_prob))

    best_hr = max(hr_candidates, key=lambda x: x[1]) if hr_candidates else None
    return entries, best_hr


def pitcher_props_for_team(pitcher_id, team, opp_team, pitcher_models):
    entries = []
    for stat_key, market_label in PITCHER_COUNT_STATS.items():
        r = project_count_stat(stat_key, pitcher_models[stat_key], pitcher_id, opp_team)
        if r:
            entries.append({"section": "Pitching", "player": r["player_display_name"], "player_id": int(pitcher_id),
                             "team": team, "market": market_label, "line": r["line"], "projected": r["projected"],
                             "model_over_prob": r["model_over_prob"], "model_std": r["model_std"]})
    return entries


def main():
    target_date, raw_games = get_slate_schedule()
    slate = parse_slate(raw_games)
    print(f"Target date: {target_date} -- {len(slate)} games")

    games_df = load_games()
    team_elo_preds, elo_params = elo_predictions(games_df, slate)
    print("Blending in starting-pitcher + bullpen strength...")
    elo_preds = blend_with_pitcher_strength(team_elo_preds, slate)
    attach_market_odds(slate)

    print("Fitting batter prop models on full historical data...")
    batter_models = {}
    for stat_key in BATTER_COUNT_STATS:
        hist = build_batter_prop_table(stat_key)
        model, std = prepare_count_model(hist)
        batter_models[stat_key] = {
            "model": model, "std": std,
            "own": batter_current_trailing(stat_key), "opp": batter_opponent_current_trailing(stat_key),
        }
    hr_df = build_batter_prop_table("home_runs")
    hr_df["actual"] = (hr_df["actual"] > 0).astype(float)
    hr_model = prepare_binary_model(hr_df)
    hr_own = batter_current_trailing("home_runs")
    hr_opp = batter_opponent_current_trailing("home_runs")
    pa_own = batter_current_trailing("pa_count")

    print("Fitting pitcher prop models on full historical data...")
    pitcher_models = {}
    for stat_key in PITCHER_COUNT_STATS:
        hist = build_pitcher_prop_table(stat_key)
        model, std = prepare_count_model(hist)
        pitcher_models[stat_key] = {
            "model": model, "std": std,
            "own": pitcher_current_trailing(stat_key), "opp": pitcher_opponent_current_trailing(stat_key),
        }

    team_abbrs = sorted({g["home_team"] for g in slate} | {g["away_team"] for g in slate})
    print(f"Fetching active rosters for {len(team_abbrs)} teams...")
    active_rosters = {abbr: get_active_roster_ids(abbr) for abbr in team_abbrs}

    games_out = []
    for i, g in enumerate(slate):
        home_batters = top_batters_for_team(pa_own, g["home_team"], active_rosters.get(g["home_team"]))
        away_batters = top_batters_for_team(pa_own, g["away_team"], active_rosters.get(g["away_team"]))

        home_props, home_best_hr = batter_props_for_team(
            g["home_team"], g["away_team"], home_batters, batter_models, hr_model, hr_own, hr_opp)
        away_props, away_best_hr = batter_props_for_team(
            g["away_team"], g["home_team"], away_batters, batter_models, hr_model, hr_own, hr_opp)

        props = home_props + away_props
        combo = None
        if home_best_hr and away_best_hr:
            p_either = 1 - (1 - home_best_hr[1]) * (1 - away_best_hr[1])
            combo = {
                "player_a": home_best_hr[0], "prob_a": round(home_best_hr[1], 3),
                "player_b": away_best_hr[0], "prob_b": round(away_best_hr[1], 3),
                "prob_either": round(p_either, 3),
            }

        if g["home_probable_pitcher"]:
            props += pitcher_props_for_team(g["home_probable_pitcher"]["id"], g["home_team"], g["away_team"], pitcher_models)
        if g["away_probable_pitcher"]:
            props += pitcher_props_for_team(g["away_probable_pitcher"]["id"], g["away_team"], g["home_team"], pitcher_models)

        elo_home = float(elo_preds[i])
        market = g["market"]
        good_value_home = good_value_away = None
        if market and market.get("home_fair_prob") is not None:
            good_value_home = elo_home > market["home_fair_prob"]
            good_value_away = (1 - elo_home) > (1 - market["home_fair_prob"])

        games_out.append({
            "awayAbbr": g["away_team"], "homeAbbr": g["home_team"],
            "awayName": g["away_name"], "homeName": g["home_name"],
            "gameDatetime": g["game_datetime"],
            "gamePk": g["game_pk"],
            "awayProbablePitcher": g["away_probable_pitcher"]["fullName"] if g["away_probable_pitcher"] else None,
            "awayProbablePitcherId": g["away_probable_pitcher"]["id"] if g["away_probable_pitcher"] else None,
            "homeProbablePitcher": g["home_probable_pitcher"]["fullName"] if g["home_probable_pitcher"] else None,
            "homeProbablePitcherId": g["home_probable_pitcher"]["id"] if g["home_probable_pitcher"] else None,
            "elo_home_prob": round(elo_home, 4),
            "team_elo_home_prob": round(float(team_elo_preds[i]), 4),
            "market": market,
            "good_value_home": good_value_home,
            "good_value_away": good_value_away,
            "hr_combo": combo,
            "props": props,
        })
        print(f"  {g['away_team']} @ {g['home_team']}: model_home={elo_preds[i]:.3f} (team-only elo={team_elo_preds[i]:.3f}) market={'yes' if g['market'] else 'no'} props={len(props)}")

    attach_featured_prop_odds(games_out, limit=int(os.environ.get("FEATURED_PROPS_LIMIT", "2")))

    pitcher_model_path = ROOT / "notebooks_out" / "mlb_pitcher_model_backtest.json"
    pitcher_blend_used = pitcher_model_path.exists()
    payload = {
        "date": target_date, "elo_params": elo_params,
        "pitcher_blend_used": pitcher_blend_used,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "games": games_out,
    }
    out_path = DATA_DIR / "dashboard_current_slate.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Generate REAL projections for the next day's MLB slate:
- Elo win probability, ratings carried forward from all completed games.
- Real DraftKings game-line odds (moneyline/run-line/total) via The Odds API.
- Player props for each team's probable starting pitcher (strikeouts) and
  the 3 batters with the highest recent plate-appearance volume per team
  (a proxy for "everyday player" -- we don't have a confirmed lineup source),
  fit on the full historical dataset since the target date is genuinely in
  the future.
- An either/or anytime-HR "special": P(at least one of two players homers),
  computed from our own anytime-HR model for the single best HR bet on each
  team, assuming independence between the two players.
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
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.games import load_games
from pipeline.mlb.elo_model import run_elo
from pipeline.mlb.team_map import full_name_to_statcast
from pipeline.mlb.props.prop_data import build_batter_prop_table, build_pitcher_prop_table
from pipeline.mlb.props.current_state import (
    batter_current_trailing, batter_opponent_current_trailing,
    pitcher_current_trailing, pitcher_opponent_current_trailing,
)
from pipeline.mlb.props.prop_models import FEATURES, over_prob
from pipeline.common.odds_api import get_game_odds, get_event_player_props

DATA_DIR = ROOT / "data" / "mlb"


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


PROP_MARKET_KEYS = {
    "Hits": "batter_hits",
    "Total Bases": "batter_total_bases",
    "Anytime HR": "batter_home_runs",
    "Pitcher Strikeouts": "pitcher_strikeouts",
}


def _norm_name(name):
    """Accent-insensitive lowercase match key (DK strips accents: Jose Ramirez)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c)).lower().strip()


def attach_featured_prop_odds(games_out, limit=1):
    """Fetch real DraftKings player-prop odds for the most competitive
    not-yet-started game(s) only -- each per-game props call costs ~4 credits,
    so the free tier can afford 1-2 games per refresh, not the slate."""
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
            # Re-aim the model probability at DraftKings' actual line instead
            # of our trailing-average proxy (binary Anytime HR already IS
            # P(over 0.5), so only count props need recomputing).
            if p["market"] != "Anytime HR" and dk.get("line") is not None and "model_std" in p:
                p["model_over_prob"] = round(float(over_prob(p["projected"], p["model_std"], dk["line"])), 3)
                p["line"] = dk["line"]
            matched += 1

        g["featured_props"] = True
        print(f"  featured prop odds: {g['awayAbbr']} @ {g['homeAbbr']} ({matched} props matched to DK)")


def fit_yardage_model(hist_df):
    model = RidgeCV(alphas=np.logspace(-1, 3, 25))
    model.fit(hist_df[FEATURES].values, hist_df["actual"].values)
    resid_std = float(np.std(hist_df["actual"].values - model.predict(hist_df[FEATURES].values)))
    return model, max(resid_std, 1e-6)


def fit_binary_model(hist_df):
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
    model.fit(hist_df[FEATURES].values, hist_df["actual"].values)
    return model


def project_count_stat(model, resid_std, own_avg, opp_avg):
    pred_mean = float(model.predict([[own_avg, opp_avg]])[0])
    line = round(own_avg * 2) / 2
    p_over = float(over_prob(pred_mean, resid_std, line))
    return round(pred_mean, 1), line, round(p_over, 3)


def top_batters_for_team(pa_own, team, n=3):
    team_players = pa_own[pa_own["team"] == team].sort_values("current_avg", ascending=False)
    return team_players.head(n).index.tolist()


def batter_props_for_team(team, opp_team, player_ids, models, own_lookups, opp_lookups):
    hits_model, hits_std, tb_model, tb_std, hr_model = models
    hits_own, tb_own, hr_own = own_lookups
    hits_opp, tb_opp, hr_opp = opp_lookups
    entries = []
    hr_candidates = []

    if opp_team not in hits_opp.index or opp_team not in tb_opp.index or opp_team not in hr_opp.index:
        return entries, None

    for pid in player_ids:
        if pid not in hits_own.index:
            continue
        name = hits_own.loc[pid, "player_display_name"]

        own_hits = hits_own.loc[pid, "current_avg"]
        if pd.notna(own_hits):
            proj, line, p_over = project_count_stat(hits_model, hits_std, float(own_hits), float(hits_opp.loc[opp_team]))
            entries.append({"player": name, "team": team, "market": "Hits", "line": line, "projected": proj,
                            "model_over_prob": p_over, "model_std": round(hits_std, 3)})

        own_tb = tb_own.loc[pid, "current_avg"] if pid in tb_own.index else None
        if own_tb is not None and pd.notna(own_tb):
            proj, line, p_over = project_count_stat(tb_model, tb_std, float(own_tb), float(tb_opp.loc[opp_team]))
            entries.append({"player": name, "team": team, "market": "Total Bases", "line": line, "projected": proj,
                            "model_over_prob": p_over, "model_std": round(tb_std, 3)})

        own_hr = hr_own.loc[pid, "current_avg"] if pid in hr_own.index else None
        if own_hr is not None and pd.notna(own_hr):
            hr_prob = float(hr_model.predict_proba([[float(own_hr), float(hr_opp.loc[opp_team])]])[:, 1][0])
            entries.append({"player": name, "team": team, "market": "Anytime HR", "model_prob": round(hr_prob, 3)})
            hr_candidates.append((name, hr_prob))

    best_hr = max(hr_candidates, key=lambda x: x[1]) if hr_candidates else None
    return entries, best_hr


def pitcher_prop(pitcher_id, opp_team, model, resid_std, own, opp):
    if pitcher_id not in own.index or opp_team not in opp.index:
        return None
    own_avg = own.loc[pitcher_id, "current_avg"]
    if pd.isna(own_avg):
        return None
    name = own.loc[pitcher_id, "player_display_name"]
    proj, line, p_over = project_count_stat(model, resid_std, float(own_avg), float(opp.loc[opp_team]))
    return {"player": name, "market": "Pitcher Strikeouts", "line": line, "projected": proj,
            "model_over_prob": p_over, "model_std": round(resid_std, 3)}


def main():
    target_date, raw_games = get_slate_schedule()
    slate = parse_slate(raw_games)
    print(f"Target date: {target_date} -- {len(slate)} games")

    games_df = load_games()
    elo_preds, elo_params = elo_predictions(games_df, slate)
    attach_market_odds(slate)

    print("Fitting prop models on full historical data...")
    hits_model, hits_std = fit_yardage_model(build_batter_prop_table("hits"))
    tb_model, tb_std = fit_yardage_model(build_batter_prop_table("total_bases"))
    pk_model, pk_std = fit_yardage_model(build_pitcher_prop_table())
    hr_df = build_batter_prop_table("home_runs")
    hr_df["actual"] = (hr_df["actual"] > 0).astype(float)
    hr_model = fit_binary_model(hr_df)

    pa_own = batter_current_trailing("pa_count")
    hits_own = batter_current_trailing("hits")
    tb_own = batter_current_trailing("total_bases")
    hr_own = batter_current_trailing("home_runs")
    hits_opp = batter_opponent_current_trailing("hits")
    tb_opp = batter_opponent_current_trailing("total_bases")
    hr_opp = batter_opponent_current_trailing("home_runs")

    models = (hits_model, hits_std, tb_model, tb_std, hr_model)
    own_lookups = (hits_own, tb_own, hr_own)
    opp_lookups = (hits_opp, tb_opp, hr_opp)
    pk_own = pitcher_current_trailing()
    pk_opp = pitcher_opponent_current_trailing()

    games_out = []
    for i, g in enumerate(slate):
        home_batters = top_batters_for_team(pa_own, g["home_team"])
        away_batters = top_batters_for_team(pa_own, g["away_team"])

        home_props, home_best_hr = batter_props_for_team(
            g["home_team"], g["away_team"], home_batters, models, own_lookups, opp_lookups)
        away_props, away_best_hr = batter_props_for_team(
            g["away_team"], g["home_team"], away_batters, models, own_lookups, opp_lookups)

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
            r = pitcher_prop(g["home_probable_pitcher"]["id"], g["away_team"], pk_model, pk_std, pk_own, pk_opp)
            if r:
                r["team"] = g["home_team"]
                props.append(r)
        if g["away_probable_pitcher"]:
            r = pitcher_prop(g["away_probable_pitcher"]["id"], g["home_team"], pk_model, pk_std, pk_own, pk_opp)
            if r:
                r["team"] = g["away_team"]
                props.append(r)

        games_out.append({
            "awayAbbr": g["away_team"], "homeAbbr": g["home_team"],
            "awayName": g["away_name"], "homeName": g["home_name"],
            "gameDatetime": g["game_datetime"],
            "awayProbablePitcher": g["away_probable_pitcher"]["fullName"] if g["away_probable_pitcher"] else None,
            "homeProbablePitcher": g["home_probable_pitcher"]["fullName"] if g["home_probable_pitcher"] else None,
            "elo_home_prob": round(float(elo_preds[i]), 4),
            "market": g["market"],
            "hr_combo": combo,
            "props": props,
        })
        print(f"  {g['away_team']} @ {g['home_team']}: elo_home={elo_preds[i]:.3f} market={'yes' if g['market'] else 'no'} props={len(props)}")

    attach_featured_prop_odds(games_out, limit=int(os.environ.get("FEATURED_PROPS_LIMIT", "1")))

    payload = {
        "date": target_date, "elo_params": elo_params,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "games": games_out,
    }
    out_path = DATA_DIR / "dashboard_current_slate.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

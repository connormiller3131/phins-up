"""Experiment: does shrinking own_trailing_avg toward a walk-forward-safe
league-average rate (weighted by how many games actually fed the trailing
average) improve MLB player-prop accuracy?

Motivation: own_trailing_avg is a single rolling mean with no signal for how
many games produced it -- a player on their 5th game back from a long
absence (MIN_GAMES floor) and one on their 15th (full WINDOW) get treated
identically by the model, even though the 5-game figure is a noisier
estimate of true talent. This is a real, quantifiable recency-bias risk
(reported by the user re: a small-sample callup, Tyler Locklear, dominating
the daily parlay), separate from and additional to the stint-pooling bug
already fixed in current_state.py.

shrunk = (n * own_trailing_avg + k * league_avg_asof_date) / (n + k)

n = games actually in the trailing window (<= WINDOW, >= MIN_GAMES by
construction). k = shrinkage strength in games-equivalent weight on the
league prior -- k=0 reproduces the current, unshrunk baseline exactly.
league_avg_asof_date is the mean of `actual` over all rows strictly before
the test date (no leakage), same walk-forward discipline as everything else
in this pipeline.

Same test window and metrics as backtest_props.py, so results are directly
comparable to the existing baseline numbers in notebooks_out/mlb_props_backtest.json.
Deploys nothing -- reports honest results only, matching the pattern used
for the NFL offense/defense feature blend (kept as documented research,
not wired in, because it didn't actually help)."""
import sys
import pathlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.player_names import get_name_lookup
from pipeline.mlb.props.prop_data import WINDOW, MIN_GAMES
from pipeline.mlb.props.prop_models import over_prob
from pipeline.common.metrics import brier_score, log_loss, calibration_curve, accuracy

DATA_DIR = ROOT / "data" / "mlb"
TEST_START = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2025-09-30")
K_VALUES = [0, 5, 10, 20, 30]


def _build_with_n(df, stat_col):
    """Same as prop_data.py's _build, plus own_trailing_n (games actually in
    the trailing window) needed for the shrinkage weight."""
    names = get_name_lookup()
    df = df.merge(names, on="player_id", how="left")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    grp = df.groupby("player_id")[stat_col]
    df["own_trailing_avg"] = grp.transform(lambda s: s.shift(1).rolling(window=WINDOW, min_periods=MIN_GAMES).mean())
    df["own_trailing_n"] = grp.transform(lambda s: s.shift(1).rolling(window=WINDOW, min_periods=MIN_GAMES).count())

    allowed = (
        df.groupby(["opponent_team", "game_date"])[stat_col]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team", stat_col: "allowed_that_game"})
        .sort_values(["defense_team", "game_date"])
    )
    allowed["opp_allowed_trailing_avg"] = allowed.groupby("defense_team")["allowed_that_game"].transform(
        lambda s: s.shift(1).rolling(window=WINDOW, min_periods=MIN_GAMES).mean()
    )
    df = df.merge(
        allowed[["defense_team", "game_date", "opp_allowed_trailing_avg"]],
        left_on=["opponent_team", "game_date"], right_on=["defense_team", "game_date"], how="left",
    )

    keep = ["player_id", "player_display_name", "team", "opponent_team", "game_date",
            stat_col, "own_trailing_avg", "own_trailing_n", "opp_allowed_trailing_avg"]
    out = df[keep].rename(columns={stat_col: "actual"})
    out = out.dropna(subset=["own_trailing_avg", "opp_allowed_trailing_avg", "player_display_name"])
    return out.reset_index(drop=True)


def add_shrunk_features(df):
    """Adds one shrunk_avg_k{k} column per K_VALUES, using a walk-forward
    league average (expanding mean of `actual` strictly before each row's
    date -- computed once via a sorted cumulative mean, not per test date,
    since it's the same value for every row sharing a date either way)."""
    df = df.sort_values("game_date").reset_index(drop=True)
    daily_sum = df.groupby("game_date")["actual"].sum()
    daily_count = df.groupby("game_date")["actual"].count()
    cum_sum_before = daily_sum.cumsum().shift(1).reindex(df["game_date"]).values
    cum_count_before = daily_count.cumsum().shift(1).reindex(df["game_date"]).values
    league_avg = cum_sum_before / cum_count_before
    df["league_avg_asof"] = league_avg

    n = df["own_trailing_n"].values
    own = df["own_trailing_avg"].values
    for k in K_VALUES:
        if k == 0:
            df[f"shrunk_avg_k{k}"] = own
        else:
            df[f"shrunk_avg_k{k}"] = (n * own + k * league_avg) / (n + k)
    return df.dropna(subset=["league_avg_asof"]).reset_index(drop=True)


def walk_forward_count_stat_feat(df, test_dates, feature_col):
    n = len(df)
    model_pred = np.full(n, np.nan)
    resid_std = np.full(n, np.nan)
    features = [feature_col, "opp_allowed_trailing_avg"]

    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < 200:
            continue
        X_train = train[features].values
        y_train = train["actual"].values
        model = RidgeCV(alphas=np.logspace(-1, 3, 25))
        model.fit(X_train, y_train)
        train_resid_std = float(np.std(y_train - model.predict(X_train)))

        X_test = df.loc[target_idx, features].values
        preds = model.predict(X_test)
        pos = df.index.get_indexer(target_idx)
        model_pred[pos] = preds
        resid_std[pos] = max(train_resid_std, 1e-6)

    return model_pred, resid_std


def walk_forward_binary_stat_feat(df, test_dates, feature_col):
    n = len(df)
    model_pred = np.full(n, np.nan)
    features = [feature_col, "opp_allowed_trailing_avg"]

    for d in test_dates:
        train = df[df["game_date"] < d]
        target_idx = df.index[df["game_date"] == d]
        if len(train) < 200 or train["actual"].nunique() < 2:
            continue
        X_train = train[features].values
        y_train = train["actual"].values
        model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=5, max_iter=2000, scoring="neg_log_loss")
        model.fit(X_train, y_train)

        X_test = df.loc[target_idx, features].values
        preds = model.predict_proba(X_test)[:, 1]
        pos = df.index.get_indexer(target_idx)
        model_pred[pos] = preds

    return model_pred


def backtest_count_stat(df, label):
    df = add_shrunk_features(df)
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    test_mask = df["game_date"].isin(test_dates)

    print(f"\n--- {label} ---")
    results = {}
    for k in K_VALUES:
        feat = f"shrunk_avg_k{k}"
        model_pred, resid_std = walk_forward_count_stat_feat(df, test_dates, feat)
        valid = test_mask.values & ~np.isnan(model_pred)
        actual = df["actual"].values[valid]
        mp, rs = model_pred[valid], resid_std[valid]

        rmse = float(np.sqrt(np.mean((actual - mp) ** 2)))
        mae = float(np.mean(np.abs(actual - mp)))
        naive_line_pred = df["own_trailing_avg"].values[valid]
        line = np.round(naive_line_pred * 2) / 2.0
        over_actual = (actual > line).astype(float)
        model_p_over = over_prob(mp, rs, line)
        brier = brier_score(over_actual, model_p_over)
        results[k] = {"n": int(valid.sum()), "rmse": rmse, "mae": mae, "proxy_line_brier": brier}
        print(f"  k={k:>3}  n={valid.sum():>5}  RMSE={rmse:.4f}  MAE={mae:.4f}  proxy-line Brier={brier:.4f}")
    return results


def backtest_anytime_hr():
    df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    df = _build_with_n(df, "home_runs")
    df["actual"] = (df["actual"] > 0).astype(float)
    df = add_shrunk_features(df)
    test_dates = sorted(df[(df["game_date"] >= TEST_START) & (df["game_date"] <= TEST_END)]["game_date"].unique())
    test_mask = df["game_date"].isin(test_dates)

    print("\n--- anytime_home_run ---")
    results = {}
    for k in K_VALUES:
        feat = f"shrunk_avg_k{k}"
        model_pred = walk_forward_binary_stat_feat(df, test_dates, feat)
        valid = test_mask.values & ~np.isnan(model_pred)
        actual = df["actual"].values[valid]
        mp = model_pred[valid]
        brier = brier_score(actual, mp)
        logloss = log_loss(actual, mp)
        acc = accuracy(actual, mp)
        results[k] = {"n": int(valid.sum()), "brier": brier, "logloss": logloss, "accuracy": acc}
        print(f"  k={k:>3}  n={valid.sum():>5}  Brier={brier:.4f}  LogLoss={logloss:.4f}  Accuracy={acc:.4f}")
    return results


def main():
    print("=== Shrinkage-toward-league-average experiment, test window 2025-08-01 to 2025-09-30 ===")
    print("k=0 is the exact current production behavior (no shrinkage) -- the baseline to beat.\n")

    all_results = {}
    batter_df = pd.read_parquet(DATA_DIR / "batter_game_logs.parquet")
    pitcher_df = pd.read_parquet(DATA_DIR / "pitcher_game_logs.parquet")

    all_results["hits"] = backtest_count_stat(_build_with_n(batter_df, "hits"), "hits (batter)")
    all_results["total_bases"] = backtest_count_stat(_build_with_n(batter_df, "total_bases"), "total_bases (batter)")
    all_results["pitcher_strikeouts"] = backtest_count_stat(_build_with_n(pitcher_df, "strikeouts"), "pitcher_strikeouts")
    all_results["anytime_home_run"] = backtest_anytime_hr()

    out_path = ROOT / "notebooks_out" / "mlb_shrinkage_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

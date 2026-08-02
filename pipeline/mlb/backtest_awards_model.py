"""Does a real stats-to-vote-share model actually pick real MVP/Cy Young
winners? Walk-forward by year (train on years strictly before the test year,
same no-leakage discipline as every other backtest here), using the real
Lahman vote-share data (pipeline/mlb/awards_data.py) -- checks whether the
model's top pick per league-year matches the real winner, not just whether
the continuous vote-share number is close.

Real, confirmed on real data before backtesting anything: Kris Bryant/Mike
Trout (2016 MVP) and Max Scherzer/Rick Porcello (2016 Cy Young) all correctly
show the highest real vote_share in their league-year in the raw data itself
-- this backtest checks whether a MODEL trained on years before 2016 can
recover that same real result without having seen it.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.mlb.awards_data import (
    build_batter_season_table, build_pitcher_season_table, attach_vote_share,
    BATTER_FEATURES, PITCHER_FEATURES,
)

TEST_YEARS = list(range(2000, 2017))  # real years with plenty of real prior-year training data


def walk_forward(table, features, min_train_rows=1000):
    table = table.dropna(subset=features + ["vote_share"]).reset_index(drop=True)
    n = len(table)
    preds = np.full(n, np.nan)
    for year in TEST_YEARS:
        train = table[table["yearID"] < year]
        test_idx = table.index[table["yearID"] == year]
        if len(train) < min_train_rows or len(test_idx) == 0:
            continue
        model = RidgeCV(alphas=np.logspace(-2, 3, 25))
        model.fit(train[features].values, train["vote_share"].values)
        p = model.predict(table.loc[test_idx, features].values)
        preds[table.index.get_indexer(test_idx)] = np.clip(p, 0, 1)
    return table, preds


def evaluate(table, preds, label):
    mask = ~np.isnan(preds)
    sub = table[mask].copy()
    sub["pred"] = preds[mask]

    rmse = float(np.sqrt(np.mean((sub["pred"] - sub["vote_share"]) ** 2)))
    print(f"\n--- {label} ---")
    print(f"Evaluated {mask.sum()} player-seasons across {sub['yearID'].nunique()} years.")
    print(f"RMSE (vote_share): {rmse:.4f}")

    correct, total, spearmans = 0, 0, []
    for (year, lg), grp in sub.groupby(["yearID", "lgID"]):
        if grp["vote_share"].max() == 0:
            continue  # no real award data for this league-year in this dataset
        real_winner = grp.loc[grp["vote_share"].idxmax(), "playerID"]
        model_pick = grp.loc[grp["pred"].idxmax(), "playerID"]
        total += 1
        correct += int(real_winner == model_pick)
        if grp["vote_share"].nunique() > 1 and grp["pred"].nunique() > 1:
            rho, _ = spearmanr(grp["vote_share"], grp["pred"])
            if not np.isnan(rho):
                spearmans.append(rho)

    print(f"Top-pick accuracy: {correct}/{total} ({correct/total:.1%}) real league-year winners correctly picked as the model's #1.")
    print(f"Mean within-league-year Spearman rank correlation: {np.mean(spearmans):.3f}")


def main():
    bat = attach_vote_share(build_batter_season_table(), "MVP")
    bat_table, bat_preds = walk_forward(bat, BATTER_FEATURES)
    evaluate(bat_table, bat_preds, "MVP (batters)")

    pit = attach_vote_share(build_pitcher_season_table(), "Cy Young")
    pit_table, pit_preds = walk_forward(pit, PITCHER_FEATURES)
    evaluate(pit_table, pit_preds, "Cy Young (pitchers)")


if __name__ == "__main__":
    main()

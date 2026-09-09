"""Does knowing WHO ELSE IS OUT improve a volume projection?

THE PROBLEM. own_trailing_avg is a player's rolling mean, and it silently
blends two different situations: games where the starter ahead of him played,
and games where that starter was out and he inherited the whole role. Measured
on 2019-2025, a backup RB averages 4.36 carries when RB1 plays and 8.51 when
RB1 is ruled Out. The model currently cannot tell those weeks apart, so it is
wrong in BOTH directions -- it under-projects him when the starter is out, and
over-projects him on normal weeks because his average carries the inherited
games inside it.

THE CANDIDATE FEATURE is `vacated_share`: the fraction of a team's recent
position-group opportunity belonging to players ruled Out this week. Zero on a
normal week, and rising as the depth chart above a player empties out. Paired
with `own_trailing_share` (how much of the group's work is already his), that
is enough for the model to learn "when 40% of this backfield is unavailable,
the remaining backs absorb it in proportion to their existing role".

Deliberately projecting OPPORTUNITY (carries, targets), not yards. Volume
redistributes; efficiency does not transfer nearly as cleanly -- a backup
handed 15 carries does not inherit the starter's yards per carry. Volume feeds
the existing yardage model, which is where efficiency already lives.

WHAT WOULD MAKE THIS NOT WORTH SHIPPING. A lead player is ruled Out in only
~3% of team-weeks, so a whole-population average will drown the effect. The
bar is therefore two-sided and reported that way below: the feature must beat
the baseline on AFFECTED weeks without degrading UNAFFECTED ones. Improving
the rare case by making the common case worse is a bad trade at this
frequency.

VERDICT: NOT SHIPPED. Run 2026-09-09 on 2022-2025 walk-forward.

  RB carries      linear      all -1.9%   affected(>5%) -0.6%   unaffected -2.0%
                  proportional            affected      +0.6%
  WR/TE targets   linear                  affected      +1.7%
                  proportional            affected      +4.4%
  RB targets      linear                  affected      +2.9%
                  proportional            affected      +9.3%

Stratified by how much of the group is actually missing, the only cell that
improves is RB carries with a genuine starter out (>30% vacated): -1.7% MAE on
196 rows across four seasons, roughly 49 a year. Receiving is consistently
WORSE, -2.5% to -2.7% at the same vacancy levels. own_trailing_share on its own
is 0.3-0.6%, which is noise for a change of this size.

WHY THE DESCRIPTIVE EFFECT DOES NOT BECOME A PREDICTIVE ONE. The 1.95x is real
and reproducible; it just is not usable:
  * own_trailing_avg already absorbs much of it. The window is six games, so a
    backup who covered for an injured starter recently already carries those
    games in his average.
  * Knowing the work doubles does not say WHO takes it. A committee splits it
    three ways and a bell-cow backfield does not, and share history does not
    separate those in advance.
  * The variance swamps it. Baseline MAE on those weeks is ~4.6 carries; a
    1.7% improvement is 0.08 of a carry.
  * The sample is tiny. ~3% of team-weeks, so there is little to fit on and
    little to gain.

WHAT ACTUALLY MATTERED was upstream and is already shipped: pulling the
current season's injury report at all, so players ruled Out are dropped rather
than projected. That is a data fix with no modelling risk, and it is worth
more than anything measured here.

Kept as a documented negative result rather than deleted, the same way the
MLB pitch-style-similarity experiment is recorded in the site footer. Re-run
it before anyone proposes this again.

    python -m pipeline.nfl.props.backtest_injury_redistribution
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.nfl.props.prop_data import build_prop_table  # noqa: E402
from pipeline.nfl.props.prop_models import FEATURES  # noqa: E402

DATA_DIR = ROOT / "data" / "nfl"
TEST_SEASONS = [2022, 2023, 2024, 2025]
EXTRA = ["own_trailing_share", "vacated_share"]


def load_out_map():
    """{(season, week, player_id)} for everyone ruled Out."""
    inj = (pl.read_parquet(DATA_DIR / "injuries.parquet")
           .filter(pl.col("report_status") == "Out")
           .filter(pl.col("gsis_id").is_not_null())
           .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
           .select(["season", "week", "gsis_id"]).unique())
    return {(r[0], r[1], r[2]) for r in inj.iter_rows()}


def add_availability_features(df, out_keys, volume_col):
    """Attach own_trailing_share and vacated_share.

    A player ruled Out has NO stat line that week, so his own row is absent
    from df -- which is exactly why the share has to be built from each
    player's TRAILING volume rather than from the current week's box score.
    Every player who appeared for the team recently carries a trailing volume
    forward, and that is what gets marked available or not.
    """
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    df["is_out"] = [
        (s, w, p) in out_keys
        for s, w, p in zip(df["season"], df["week"], df["player_id"])
    ]

    # Roster-carrying: for each team-week, every player with a trailing volume
    # who has played for this team this season, whether or not he suited up.
    vol = df[["season", "week", "team", "player_id", volume_col]].copy()
    vol["trailing_vol"] = (
        vol.sort_values(["player_id", "season", "week"])
           .groupby("player_id")[volume_col]
           .transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    )
    roster = vol.dropna(subset=["trailing_vol"])

    # Carry each player's latest trailing volume forward across his team's
    # weeks, so a player who is Out (and therefore missing from that week)
    # still contributes the share he is vacating.
    frames = []
    for (season, team), grp in roster.groupby(["season", "team"], sort=False):
        weeks = sorted(grp["week"].unique())
        for wk in weeks:
            hist = grp[grp["week"] <= wk].sort_values("week")
            latest = hist.groupby("player_id")["trailing_vol"].last()
            if latest.empty or latest.sum() <= 0:
                continue
            share = latest / latest.sum()
            is_out = pd.Series(
                [(season, wk, pid) in out_keys for pid in share.index], index=share.index)
            frames.append(pd.DataFrame({
                "season": season, "week": wk, "team": team,
                "player_id": share.index,
                "own_trailing_share": share.values,
                "vacated_share": float(share[is_out].sum()),
            }))
    if not frames:
        raise SystemExit("no roster shares built -- check volume column")
    shares = pd.concat(frames, ignore_index=True)

    out = df.merge(shares, on=["season", "week", "team", "player_id"], how="left")
    out["own_trailing_share"] = out["own_trailing_share"].fillna(0.0)
    # A player's OWN absence is not opportunity he can absorb, so remove it.
    out["vacated_share"] = (out["vacated_share"].fillna(0.0)
                            - np.where(out["is_out"], out["own_trailing_share"], 0.0)).clip(lower=0.0)
    return out


def walk_forward(df, features):
    """Same protocol as prop_models.walk_forward_yardage: refit weekly on
    everything strictly before that week, so nothing sees its own future."""
    pred = np.full(len(df), np.nan)
    keys = (df.loc[df["season"].isin(TEST_SEASONS), ["season", "week"]]
              .drop_duplicates().sort_values(["season", "week"]))
    for season, week in keys.itertuples(index=False):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        idx = df.index[(df["season"] == season) & (df["week"] == week)]
        if len(train) < 200 or not len(idx):
            continue
        m = RidgeCV(alphas=np.logspace(-1, 3, 25))
        m.fit(train[features].values, train["actual"].values)
        pred[df.index.get_indexer(idx)] = m.predict(df.loc[idx, features].values)
    return pred


def report(label, df, base, cand):
    ok = ~np.isnan(base) & ~np.isnan(cand)
    d = df[ok]
    b, c = base[ok], cand[ok]
    affected = d["vacated_share"].values > 0.05

    print(f"\n=== {label} ===")
    print(f"{'subset':<26}{'n':>7}{'MAE base':>11}{'MAE +inj':>11}{'change':>10}")
    for name, mask in (("all rows", np.ones(len(d), bool)),
                       ("AFFECTED (vacated>5%)", affected),
                       ("unaffected", ~affected)):
        if mask.sum() == 0:
            continue
        mb = np.abs(b[mask] - d["actual"].values[mask]).mean()
        mc = np.abs(c[mask] - d["actual"].values[mask]).mean()
        pct = (mc - mb) / mb * 100
        print(f"{name:<26}{mask.sum():>7}{mb:>11.3f}{mc:>11.3f}{pct:>9.1f}%")
    return d, b, c, affected


def run(stat_col, positions, volume_col, label):
    df = build_prop_table(stat_col, positions, volume_col=volume_col)
    df = add_availability_features(df, load_out_map(), volume_col="actual")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    base = walk_forward(df, FEATURES)
    cand = walk_forward(df, FEATURES + EXTRA)
    report(label + "  [linear: +vacated_share as a feature]", df, base, cand)

    # PROPORTIONAL FORM. Redistribution is multiplicative, not additive: if 40%
    # of a backfield is unavailable, the remaining backs split the same work
    # among 60% as much player, so each one's expected share scales by
    # 1/(1-vacated). A linear Ridge term cannot express that, which makes the
    # arm above a weak test of the actual mechanism rather than of the idea.
    prop = df.copy()
    scale = 1.0 / (1.0 - prop["vacated_share"].clip(upper=0.75))
    prop["own_trailing_avg"] = prop["own_trailing_avg"] * scale
    if "own_trailing_volume" in prop.columns:
        prop["own_trailing_volume"] = prop["own_trailing_volume"] * scale
    cand2 = walk_forward(prop, FEATURES)
    return report(label + "  [proportional: scale trailing by 1/(1-vacated)]",
                  df, base, cand2)


def main():
    print("Walk-forward, refit weekly, test seasons", TEST_SEASONS)
    print("Baseline = current FEATURES. Candidate = FEATURES +", EXTRA)
    run("carries", ["RB"], "carries", "RB carries")
    run("targets", ["WR", "TE"], "targets", "WR/TE targets")
    run("targets", ["RB"], "targets", "RB targets")


if __name__ == "__main__":
    main()

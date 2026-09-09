"""Who else is unavailable, expressed as two features.

    own_trailing_share  this player's share of his team's recent opportunity
                        in his position group
    vacated_share       the share belonging to teammates ruled Out this week,
                        excluding the player's own absence

NARROW ON PURPOSE. These are wired into ONE market, RB Carries, because that
is the only place a walk-forward backtest supported them
(backtest_injury_redistribution.py, run 2026-09-09):

    RB carries        vacated <30%  -1.9%   vacated >30%  -1.7%   SHIP
    RB rushing yards  vacated <30%  -1.4%   vacated >30%  +4.7%   DO NOT
    WR/TE targets                          affected      +1.7%   DO NOT
    RB targets                             affected      +2.9%   DO NOT

Rushing yards is the instructive one: it improves on ordinary weeks and gets
materially WORSE exactly when a starter is out. Volume redistributes,
efficiency does not -- a backup handed fifteen carries does not inherit the
starter's yards per carry, and pushing his yardage up is the wrong correction.
So carries move and yards do not, which is a real asymmetry rather than an
oversight. Do not "fix" the inconsistency by extending this to yards without
re-running that backtest.

A player ruled Out has no stat line that week, so his row is simply absent
from the training frame. That is precisely why shares are built from each
player's TRAILING volume carried forward, not from the current week's box
score -- the whole point is to account for someone who is not there.
"""
import pathlib

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "nfl"

# Markets that receive these features. Adding to this set without a backtest
# is how the receiving props above would have shipped a regression.
AVAILABILITY_STATS = {"carries"}

FEATURES = ["own_trailing_share", "vacated_share"]

# Above this, the card says so. Set at the level the backtest actually
# measured a gain (>30% vacated), so the note appears exactly when the
# adjustment is doing something the evidence supports.
SURFACE_MIN = 0.30

# Window used to carry a player's usage forward. Shorter than the props'
# own WINDOW because a depth-chart role changes faster than a rate does.
SHARE_WINDOW = 6
SHARE_MIN = 2


def load_out_keys():
    """{(season, week, gsis_id)} for every player ruled Out."""
    path = DATA_DIR / "injuries.parquet"
    if not path.exists():
        return set()
    inj = (pl.read_parquet(path)
           .filter(pl.col("report_status") == "Out")
           .filter(pl.col("gsis_id").is_not_null())
           .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
           .select(["season", "week", "gsis_id"]).unique())
    return {(r[0], r[1], r[2]) for r in inj.iter_rows()}


def _team_week_shares(roster, out_keys):
    """roster: season/week/team/player_id/trailing_vol -> per-row shares."""
    frames = []
    for (season, team), grp in roster.groupby(["season", "team"], sort=False):
        for wk in sorted(grp["week"].unique()):
            hist = grp[grp["week"] <= wk].sort_values("week")
            latest = hist.groupby("player_id")["trailing_vol"].last()
            if latest.empty or latest.sum() <= 0:
                continue
            share = latest / latest.sum()
            is_out = pd.Series([(season, wk, pid) in out_keys for pid in share.index],
                               index=share.index)
            frames.append(pd.DataFrame({
                "season": season, "week": wk, "team": team,
                "player_id": share.index,
                "own_trailing_share": share.values,
                "vacated_share": float(share[is_out].sum()),
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_features(df, out_keys=None, value_col="actual"):
    """Attach both features to a prop table. Mutating-free; returns a copy.

    This is the exact routine the backtest scored, kept in one place so a
    change here cannot silently diverge from the numbers in its docstring.
    """
    out_keys = load_out_keys() if out_keys is None else out_keys
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    df["is_out"] = [(s, w, p) in out_keys
                    for s, w, p in zip(df["season"], df["week"], df["player_id"])]

    vol = df[["season", "week", "team", "player_id", value_col]].copy()
    vol["trailing_vol"] = (
        vol.sort_values(["player_id", "season", "week"])
           .groupby("player_id")[value_col]
           .transform(lambda s: s.shift(1).rolling(SHARE_WINDOW, min_periods=SHARE_MIN).mean())
    )
    shares = _team_week_shares(vol.dropna(subset=["trailing_vol"]), out_keys)
    if shares.empty:
        df["own_trailing_share"] = 0.0
        df["vacated_share"] = 0.0
        return df

    out = df.merge(shares, on=["season", "week", "team", "player_id"], how="left")
    out["own_trailing_share"] = out["own_trailing_share"].fillna(0.0)
    # A player's own absence is not opportunity he can absorb.
    out["vacated_share"] = (out["vacated_share"].fillna(0.0)
                            - np.where(out["is_out"], out["own_trailing_share"], 0.0)
                            ).clip(lower=0.0)
    return out


def live_shares(current_own, roster_ids, out_ids):
    """Shares for a week that has not been played.

    current_own: what current_state.player_current_trailing returns, indexed
                 by player_id with `team` and `current_avg`.
    roster_ids:  the players actually on this team's depth chart right now.
    out_ids:     of those, the ones ruled Out for the upcoming week.

    THE ROSTER ARGUMENT IS LOAD-BEARING. current_own holds every player who
    has ever accumulated trailing history at the position, including ones who
    left the team seasons ago -- New England's carries table still lists James
    White and D'Ernest Johnson. Dividing by all of them inflates the
    denominator and shrinks every share: measured on this week's real data,
    Henderson's vacated share came out 0.24 against a true 0.49, so the
    adjustment landed at roughly half strength.

    That also made inference disagree with training, where _team_week_shares
    is scoped per season and team and therefore only ever sees that season's
    squad. Passing the live depth chart restores the same denominator on both
    sides, which is the point: a model fed a differently-constructed feature
    than it was fitted on is not the model that was backtested.

    Uses current_avg rather than a shifted mean because for an unplayed week
    the most recent games ARE the history, matching how current_state already
    treats every other feature.
    """
    roster_ids = set(roster_ids)
    df = current_own[current_own.index.isin(roster_ids)].dropna(subset=["current_avg"])
    total = float(df["current_avg"].sum())
    if total <= 0:
        return {}
    shares = df["current_avg"] / total
    vacated = float(shares[[pid in out_ids for pid in shares.index]].sum())
    return {pid: (float(sh), max(0.0, vacated - (float(sh) if pid in out_ids else 0.0)))
            for pid, sh in shares.items()}

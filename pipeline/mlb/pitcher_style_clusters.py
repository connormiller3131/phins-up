"""Group pitchers into discrete pitching-style clusters from their actual
Statcast pitch mix and velocity (not just handedness) -- e.g. a hard-
throwing fastball/slider pitcher vs. a sinker-heavy groundball pitcher vs.
a high-spin curveball specialist. Used to ask 'how does this batter perform
against pitchers who throw like today's starter', independent of whether
they've faced that specific pitcher before.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
N_CLUSTERS = 8

# Collapse Statcast's granular pitch codes into the buckets a fan would
# actually recognize, so usage% is comparable across pitchers who classify
# similar pitches slightly differently (e.g. FF vs FA, SI vs SF).
PITCH_GROUPS = {
    "FF": "four_seam", "FA": "four_seam",
    "SI": "sinker", "FT": "sinker",
    "FC": "cutter",
    "SL": "slider", "ST": "slider", "SV": "slider",
    "CU": "curve", "KC": "curve", "CS": "curve",
    "CH": "changeup", "FS": "changeup", "FO": "changeup",
    "KN": "knuckle",
}
GROUP_COLS = ["four_seam", "sinker", "cutter", "slider", "curve", "changeup", "knuckle"]


def build_pitcher_style_vectors():
    prof = pd.read_parquet(DATA_DIR / "pitcher_pitch_profile.parquet")
    prof["group"] = prof["pitch_type"].map(PITCH_GROUPS)
    prof = prof.dropna(subset=["group"])

    # collapse the monthly chunks to one row per (pitcher, group), pitch-count weighted
    prof["_w_speed"] = prof["avg_speed"] * prof["n_pitches"]
    agg = prof.groupby(["player_id", "group"]).agg(
        n_pitches=("n_pitches", "sum"), _w_speed=("_w_speed", "sum")
    ).reset_index()
    agg["avg_speed"] = agg["_w_speed"] / agg["n_pitches"]

    totals = agg.groupby("player_id")["n_pitches"].sum().rename("total_pitches")
    agg = agg.merge(totals, on="player_id")
    agg["usage_pct"] = agg["n_pitches"] / agg["total_pitches"]

    usage = agg.pivot_table(index="player_id", columns="group", values="usage_pct", fill_value=0.0)
    usage = usage.reindex(columns=GROUP_COLS, fill_value=0.0)

    # primary-pitch velocity: the speed of whichever group each pitcher throws most
    primary = agg.loc[agg.groupby("player_id")["n_pitches"].idxmax()][["player_id", "avg_speed"]]
    primary = primary.set_index("player_id")["avg_speed"].rename("primary_velo")

    vol = agg.groupby("player_id")["total_pitches"].first()
    features = usage.join(primary).join(vol.rename("total_pitches"))
    features = features[features["total_pitches"] >= 200]  # enough pitches to trust the profile
    return features


def build_clusters(n_clusters=N_CLUSTERS, random_state=0):
    features = build_pitcher_style_vectors()
    X = features[GROUP_COLS + ["primary_velo"]].values
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    features = features.copy()
    features["style_cluster"] = labels
    return features


def describe_clusters(features):
    """Human-readable summary: dominant pitch type + avg velo per cluster."""
    out = []
    for c, grp in features.groupby("style_cluster"):
        top_pitch = grp[GROUP_COLS].mean().idxmax()
        out.append({
            "cluster": int(c), "n_pitchers": len(grp),
            "dominant_pitch": top_pitch, "avg_velo": round(float(grp["primary_velo"].mean()), 1),
        })
    return pd.DataFrame(out).sort_values("cluster")


if __name__ == "__main__":
    features = build_clusters()
    print(features.shape, "pitchers clustered")
    print(describe_clusters(features))
    features[["style_cluster"]].to_parquet(DATA_DIR / "pitcher_style_clusters.parquet")

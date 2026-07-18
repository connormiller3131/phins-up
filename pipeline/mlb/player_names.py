"""MLBAM player ID -> display name lookup, via pybaseball's Chadwick register
(cached locally to parquet so we don't re-download on every run)."""
import pathlib
import warnings
import pandas as pd
import pybaseball as pb

warnings.filterwarnings("ignore")

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mlb"
CACHE_PATH = DATA_DIR / "player_names.parquet"

_cache = None


def get_name_lookup():
    global _cache
    if _cache is not None:
        return _cache

    if CACHE_PATH.exists():
        _cache = pd.read_parquet(CACHE_PATH)
        return _cache

    reg = pb.chadwick_register()
    reg = reg[reg["key_mlbam"].notna()].copy()
    reg["player_id"] = reg["key_mlbam"].astype(int)
    reg["player_display_name"] = reg["name_first"].fillna("") + " " + reg["name_last"].fillna("")
    reg["player_display_name"] = reg["player_display_name"].str.strip()
    out = reg[["player_id", "player_display_name"]].drop_duplicates("player_id")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_PATH)
    _cache = out
    return out


if __name__ == "__main__":
    lut = get_name_lookup()
    print(lut.shape)
    print(lut.head())

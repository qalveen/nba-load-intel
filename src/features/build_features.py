"""Build the modelling table: backward-looking features + a forward label."""
import logging

import numpy as np
import pandas as pd

import config
from src.storage.duckdb_io import read_parquet, write_layer

log = logging.getLogger(__name__)

ABSENCE_DAYS = 5
FEATURES_DIR = config.DATA_DIR / "features"

FEATURE_COLS = [
    "MIN", "REST_DAYS", "IS_B2B", "ACWR", "ACUTE_LOAD", "CHRONIC_LOAD",
    "GAMES_IN_CHRONIC", "GAMES_LAST_7D", "GAMES_LAST_14D", "B2B_LAST_14D",
    "CUM_SEASON_MIN", "GAME_NUM", "AVG_MIN_SEASON", "MIN_VS_AVG",
    "AGE", "PLAYER_WEIGHT", "PLAYER_HEIGHT_INCHES",
]


def _player_season_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("GAME_DATE").copy()
    gi = g.set_index("GAME_DATE")

    g["GAMES_LAST_7D"] = gi["MIN"].rolling("7D").count().values
    g["GAMES_LAST_14D"] = gi["MIN"].rolling("14D").count().values
    g["B2B_LAST_14D"] = gi["IS_B2B"].fillna(0).rolling("14D").sum().values

    g["CUM_SEASON_MIN"] = g["MIN"].cumsum()
    g["GAME_NUM"] = np.arange(1, len(g) + 1)
    g["AVG_MIN_SEASON"] = g["CUM_SEASON_MIN"] / g["GAME_NUM"]
    g["MIN_VS_AVG"] = g["MIN"] - g["AVG_MIN_SEASON"]

    next_gap = (g["GAME_DATE"].shift(-1) - g["GAME_DATE"]).dt.days
    g["NEXT_GAP"] = next_gap
    g["TARGET"] = np.where(
        next_gap.isna(), np.nan, (next_gap >= ABSENCE_DAYS).astype(float)
    )
    return g


def run():
    gold = read_parquet(config.GOLD_DIR / "player_load.parquet")
    gold["GAME_DATE"] = pd.to_datetime(gold["GAME_DATE"])

    parts = [
        _player_season_features(g)
        for _, g in gold.groupby(["PLAYER_ID", "SEASON"], sort=False)
    ]
    tbl = pd.concat(parts, ignore_index=True)

    tbl = tbl[tbl["TARGET"].notna()].copy()
    tbl["TARGET"] = tbl["TARGET"].astype(int)

    bio_path = config.BRONZE_DIR / "player_bio.parquet"
    if bio_path.exists():
        bio = read_parquet(bio_path)
        tbl = tbl.merge(bio, on=["PLAYER_ID", "SEASON"], how="left")
        log.info("joined player bio (age); AGE null rate %.1f%%",
                 100 * tbl["AGE"].isna().mean())
    else:
        log.warning("player_bio.parquet missing -- run src.ingest.player_bio")
        for c in ["AGE", "PLAYER_WEIGHT", "PLAYER_HEIGHT_INCHES"]:
            tbl[c] = np.nan

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    write_layer(tbl, FEATURES_DIR, "model_table")
    log.info(
        "model_table: %d rows | positive (absence) rate %.1f%%",
        len(tbl), 100 * tbl["TARGET"].mean(),
    )
    return tbl


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()
    run()

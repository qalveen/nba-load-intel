"""Gold: domain-informed load metrics (acute:chronic workload ratio).

ACWR is only trustworthy once a player has had a full chronic window of
CONTINUOUS exposure since their last real break -- offseason or mid-season
injury. Both are "stints" needing a fresh warm-up.
"""
import logging

import numpy as np
import pandas as pd

import config
from src.storage.duckdb_io import read_parquet, write_layer

log = logging.getLogger(__name__)

GAP_RESET_DAYS = 14          # a layoff longer than this starts a new stint
MIN_GAMES_IN_CHRONIC = 4     # min games in trailing window to trust ACWR


def _player_season_acwr(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("GAME_DATE").copy()
    season_start = g["GAME_DATE"].min()

    gap = g["GAME_DATE"].diff().dt.days
    stint = (gap > GAP_RESET_DAYS).cumsum()
    g["STINT_START"] = g.groupby(stint)["GAME_DATE"].transform("min")

    cal = pd.DataFrame({
        "GAME_DATE": pd.date_range(season_start, g["GAME_DATE"].max(), freq="D")
    })
    daily = cal.merge(g[["GAME_DATE", "MIN"]], on="GAME_DATE", how="left")
    daily["MIN"] = daily["MIN"].fillna(0.0)
    daily = daily.set_index("GAME_DATE")

    acute = daily["MIN"].rolling(f"{config.ACUTE_WINDOW_DAYS}D").sum()
    chronic_sum = daily["MIN"].rolling(f"{config.CHRONIC_WINDOW_DAYS}D").sum()
    weeks = config.CHRONIC_WINDOW_DAYS / config.ACUTE_WINDOW_DAYS
    chronic = chronic_sum / weeks
    games_in_chronic = daily["MIN"].gt(0).rolling(
        f"{config.CHRONIC_WINDOW_DAYS}D"
    ).sum()
    acwr = np.where(chronic > 0, acute / chronic, np.nan)

    feat = pd.DataFrame(
        {
            "ACUTE_LOAD": acute,
            "CHRONIC_LOAD": chronic,
            "ACWR": acwr,
            "GAMES_IN_CHRONIC": games_in_chronic,
        },
        index=daily.index,
    ).reset_index()

    out = g.merge(feat, on="GAME_DATE", how="left")

    days_since_stint = (out["GAME_DATE"] - out["STINT_START"]).dt.days
    out.loc[days_since_stint < config.CHRONIC_WINDOW_DAYS, "ACWR"] = np.nan
    out.loc[out["GAMES_IN_CHRONIC"] < MIN_GAMES_IN_CHRONIC, "ACWR"] = np.nan

    out["REST_DAYS"] = out["GAME_DATE"].diff().dt.days
    out["IS_B2B"] = (out["REST_DAYS"] == 1).astype("Int64")
    return out.drop(columns=["STINT_START"])


def _flag(acwr):
    if pd.isna(acwr):
        return "insufficient_history"
    lo, hi = config.ACWR_SWEET_SPOT
    if acwr < lo:
        return "undertrained"
    if acwr <= hi:
        return "sweet_spot"
    if acwr < config.ACWR_DANGER:
        return "elevated"
    return "danger"


def run():
    silver = read_parquet(config.SILVER_DIR / "player_game_logs.parquet")

    parts = []
    for (player_id, season), g in silver.groupby(["PLAYER_ID", "SEASON"], sort=False):
        pg = _player_season_acwr(g)
        pg["PLAYER_ID"] = player_id
        pg["SEASON"] = season
        parts.append(pg)

    out = pd.concat(parts, ignore_index=True)
    out["LOAD_FLAG"] = out["ACWR"].map(_flag)

    write_layer(out, config.GOLD_DIR, "player_load")
    log.info("gold player_load built (%d player-game rows)", len(out))
    return out

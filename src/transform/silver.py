"""Silver: clean, type, and dedupe raw game logs."""
import logging

import pandas as pd

import config
from src.storage.duckdb_io import read_parquet, write_layer

log = logging.getLogger(__name__)


def _parse_minutes(val):
    """Normalize minutes to a float.

    LeagueGameLog gives MIN as a number, but box-score endpoints give 'MM:SS'.
    Handle both so this stays reusable when tracking data is added.
    """
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    if ":" in s:
        m, sec = s.split(":")
        return int(m) + int(sec) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def run():
    raw = read_parquet(config.BRONZE_DIR / "player_game_logs.parquet")
    df = raw.copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = df["MIN"].map(_parse_minutes)

    # A player appears at most once per game -> natural key.
    before = len(df)
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"]).reset_index(drop=True)
    if len(df) < before:
        log.info("dropped %d duplicate player-game rows", before - len(df))

    keep = [
        "SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
        "PTS", "REB", "AST", "PLUS_MINUS",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    write_layer(df, config.SILVER_DIR, "player_game_logs")
    return df

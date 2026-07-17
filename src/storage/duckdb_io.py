"""Parquet + DuckDB IO for the medallion (bronze/silver/gold) layers.

Parquet on disk keeps runs idempotent and re-runnable; DuckDB gives the
analyst/SQL layer a zero-infra warehouse over the gold tables.
"""
import logging
from pathlib import Path

import duckdb
import pandas as pd

import config

log = logging.getLogger(__name__)


def _ensure_dirs():
    for d in (config.BRONZE_DIR, config.SILVER_DIR, config.GOLD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_bronze(df: pd.DataFrame, name: str) -> Path:
    return write_layer(df, config.BRONZE_DIR, name)


def write_layer(df: pd.DataFrame, layer_dir: Path, name: str) -> Path:
    _ensure_dirs()
    path = layer_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("wrote %s/%s (%d rows)", layer_dir.name, name, len(df))
    return path


def read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_to_duckdb():
    """Register every gold parquet as a DuckDB table for SQL + the dashboard."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    tables = 0
    for pq in sorted(config.GOLD_DIR.glob("*.parquet")):
        table = pq.stem
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{pq.as_posix()}')"
        )
        log.info("loaded gold table: %s", table)
        tables += 1
    con.close()
    if tables == 0:
        log.warning("no gold parquet files found to load")

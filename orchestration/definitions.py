"""Dagster orchestration for the NBA load pipeline (Phase 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dagster import (
    asset,
    asset_check,
    AssetCheckResult,
    Definitions,
    define_asset_job,
    ScheduleDefinition,
    MetadataValue,
)

import config
from src.ingest import game_logs
from src.transform import silver, gold
from src.storage.duckdb_io import load_to_duckdb, read_parquet


@asset(group_name="etl")
def bronze_player_game_logs(context) -> None:
    """Ingest player game logs from nba_api (rate-limited)."""
    df = game_logs.run()
    context.add_output_metadata({"rows": len(df)})


@asset(group_name="etl", deps=[bronze_player_game_logs])
def silver_player_game_logs(context) -> None:
    """Clean, type, and dedupe the raw logs."""
    df = silver.run()
    context.add_output_metadata({"rows": len(df)})


@asset(group_name="etl", deps=[silver_player_game_logs])
def gold_player_load(context) -> None:
    """Compute ACWR load metrics and flags."""
    df = gold.run()
    scored = df[df["LOAD_FLAG"] != "insufficient_history"]
    danger = (scored["LOAD_FLAG"] == "danger").mean() if len(scored) else 0
    context.add_output_metadata({
        "rows": len(df),
        "danger_pct": MetadataValue.float(round(100 * float(danger), 2)),
    })


@asset(group_name="serving", deps=[gold_player_load])
def warehouse_duckdb(context) -> None:
    """Load gold parquet into DuckDB for SQL and the dashboard."""
    load_to_duckdb()


_ml_assets = []
try:
    from src.features import build_features

    @asset(group_name="ml", deps=[gold_player_load])
    def model_table(context) -> None:
        """Engineer features + injury-proxy label for the model."""
        df = build_features.run()
        context.add_output_metadata({
            "rows": len(df),
            "positive_rate": MetadataValue.float(
                round(float(df["TARGET"].mean()), 4)
            ),
        })

    _ml_assets = [model_table]
except ImportError:
    pass


@asset_check(asset=gold_player_load)
def gold_not_empty_and_keyed():
    df = read_parquet(config.GOLD_DIR / "player_load.parquet")
    ok = len(df) > 0 and df["PLAYER_ID"].notna().all()
    return AssetCheckResult(
        passed=bool(ok),
        metadata={"rows": len(df),
                  "null_player_ids": int(df["PLAYER_ID"].isna().sum())},
    )


@asset_check(asset=gold_player_load)
def acwr_within_sane_range():
    df = read_parquet(config.GOLD_DIR / "player_load.parquet")
    vals = df["ACWR"].dropna()
    ok = len(vals) > 0 and vals.between(0, 10).all()
    return AssetCheckResult(
        passed=bool(ok),
        metadata={"max_acwr": float(vals.max()) if len(vals) else 0.0,
                  "scored_games": int(len(vals))},
    )


nightly_pipeline = define_asset_job("nightly_pipeline", selection="*")

nightly_schedule = ScheduleDefinition(
    job=nightly_pipeline,
    cron_schedule="0 8 * * *",
)

defs = Definitions(
    assets=[
        bronze_player_game_logs,
        silver_player_game_logs,
        gold_player_load,
        warehouse_duckdb,
    ] + _ml_assets,
    asset_checks=[gold_not_empty_and_keyed, acwr_within_sane_range],
    jobs=[nightly_pipeline],
    schedules=[nightly_schedule],
)

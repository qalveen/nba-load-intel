"""End-to-end pipeline: bronze -> silver -> gold -> duckdb.

Run:  python -m src.pipeline

This is the local equivalent of the Dagster/Airflow DAG. In Phase 2 each
run() below becomes an orchestrated op/task with the same call order, so the
migration is mechanical.
"""
import logging

from src.ingest import game_logs
from src.transform import silver, gold
from src.storage.duckdb_io import load_to_duckdb
from src.utils.logging import setup_logging


def main():
    setup_logging()
    log = logging.getLogger("pipeline")

    log.info("STEP 1/4  bronze: ingest player game logs")
    game_logs.run()

    log.info("STEP 2/4  silver: clean + dedupe")
    silver.run()

    log.info("STEP 3/4  gold: load metrics (ACWR)")
    gold.run()

    log.info("STEP 4/4  load gold -> duckdb")
    load_to_duckdb()

    log.info("pipeline complete")


if __name__ == "__main__":
    main()

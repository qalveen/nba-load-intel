"""Score every player-game with the trained model -> risk probability.

Produces features/scored.parquet for the dashboard's predicted risk board.
Run:  python -m src.model.score   (after training)
"""
import logging

import pandas as pd
import joblib

import config
from src.storage.duckdb_io import read_parquet, write_layer
from src.features.build_features import FEATURE_COLS, FEATURES_DIR
from src.utils.logging import setup_logging

REPORTS_DIR = config.ROOT / "reports"
log = logging.getLogger("score")


def run():
    model = joblib.load(REPORTS_DIR / "gbm_model.joblib")
    tbl = read_parquet(FEATURES_DIR / "model_table.parquet")
    tbl = tbl[tbl["ACWR"].notna()].copy()

    X = tbl[FEATURE_COLS].astype(float)
    tbl["RISK_PROBA"] = model.predict_proba(X)[:, 1]

    keep = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON", "GAME_DATE",
        "MIN", "ACWR", "REST_DAYS", "LOAD_FLAG", "RISK_PROBA",
    ]
    out = tbl[[c for c in keep if c in tbl.columns]]
    write_layer(out, FEATURES_DIR, "scored")
    log.info("scored %d games | mean risk %.3f", len(out), out["RISK_PROBA"].mean())
    return out


if __name__ == "__main__":
    setup_logging()
    run()

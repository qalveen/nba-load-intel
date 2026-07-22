"""Bronze ingestion: player bio stats (age, height, weight) per season."""
import logging

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerbiostats

import config
from src.ingest.client import call_endpoint
from src.storage.duckdb_io import write_bronze

log = logging.getLogger(__name__)


def fetch_season_bio(season: str) -> pd.DataFrame:
    ep = call_endpoint(
        leaguedashplayerbiostats.LeagueDashPlayerBioStats,
        season=season,
        season_type_all_star="Regular Season",
    )
    df = ep.get_data_frames()[0]
    df["SEASON"] = season
    return df


def run(seasons=None):
    seasons = seasons or config.SEASONS
    frames = [fetch_season_bio(s) for s in seasons]
    bio = pd.concat(frames, ignore_index=True)

    keep = ["PLAYER_ID", "SEASON", "AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]
    bio = bio[[c for c in keep if c in bio.columns]]
    for c in ["AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]:
        if c in bio.columns:
            bio[c] = pd.to_numeric(bio[c], errors="coerce")

    write_bronze(bio, "player_bio")
    log.info("player_bio: %d player-season rows", len(bio))
    return bio


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()
    run()

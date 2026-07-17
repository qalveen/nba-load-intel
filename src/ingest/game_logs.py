"""Bronze ingestion: player game logs -- the load foundation.

Minutes played is our primary load proxy. LeagueGameLog gives one row per
player per game for a whole season in a single call, which keeps us well
under rate limits. Tracking data (distance, avg speed) is a Phase 2
enrichment via leaguedashptstats.
"""
import logging

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams

import config
from src.ingest.client import call_endpoint
from src.storage.duckdb_io import write_bronze

log = logging.getLogger(__name__)


def fetch_season_player_logs(season: str) -> pd.DataFrame:
    """All player-level game logs for one regular season."""
    ep = call_endpoint(
        leaguegamelog.LeagueGameLog,
        season=season,
        player_or_team_abbreviation="P",   # P = player rows (T = team rows)
        season_type_all_star="Regular Season",
    )
    df = ep.get_data_frames()[0]
    df["SEASON"] = season
    log.info("  %s: %d player-game rows", season, len(df))
    return df


def run(seasons=None, team_abbr=config.INGEST_TEAM_ABBR):
    seasons = seasons or config.SEASONS
    frames = [fetch_season_player_logs(s) for s in seasons]
    all_logs = pd.concat(frames, ignore_index=True)

    if team_abbr:
        match = [t for t in teams.get_teams() if t["abbreviation"] == team_abbr]
        if not match:
            raise ValueError(f"unknown team abbreviation: {team_abbr}")
        team_id = match[0]["id"]
        all_logs = all_logs[all_logs["TEAM_ID"] == team_id].copy()
        log.info("filtered to %s -> %d rows", team_abbr, len(all_logs))

    write_bronze(all_logs, "player_game_logs")
    return all_logs

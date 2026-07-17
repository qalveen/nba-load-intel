"""Central configuration for the NBA Load Intelligence pipeline."""
from pathlib import Path

# --- Ingestion scope ---------------------------------------------------------
# NBA season format. Start with a few seasons; scale up once the pipeline is
# proven end to end.
SEASONS = ["2022-23", "2023-24", "2024-25"]

# Start small: ingest ONE team end-to-end before scaling to the league, so you
# aren't debugging rate limits and modeling at the same time.
# Set to None to ingest all 30 teams.
INGEST_TEAM_ABBR = "LAL"

# --- Storage layout (medallion architecture) ---------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"   # raw, as-pulled
SILVER_DIR = DATA_DIR / "silver"   # cleaned, deduped, typed
GOLD_DIR = DATA_DIR / "gold"       # analytics-ready load features
DB_PATH = DATA_DIR / "nba_load.duckdb"

# --- Rate limiting -----------------------------------------------------------
# stats.nba.com throttles aggressively. These defaults are conservative.
REQUEST_DELAY_SECONDS = 0.6   # min gap between calls
MAX_RETRIES = 5
BACKOFF_BASE = 2.0            # exponential backoff base (2^attempt seconds)
REQUEST_TIMEOUT = 30

# --- Load model params -------------------------------------------------------
# Acute:Chronic Workload Ratio (ACWR) is a real sports-science concept:
# short-term load vs. rolling baseline. Sweet spot ~0.8-1.3; >1.5 is elevated
# injury risk in the literature.
ACUTE_WINDOW_DAYS = 7
CHRONIC_WINDOW_DAYS = 28
ACWR_SWEET_SPOT = (0.8, 1.3)
ACWR_DANGER = 1.5

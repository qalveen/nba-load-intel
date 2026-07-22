"""Central configuration for the NBA Load Intelligence pipeline."""
from datetime import date
from pathlib import Path

# --- Ingestion scope ---------------------------------------------------------
# Seasons are computed from today's date, so the pipeline always includes the
# current/most-recent season automatically. An NBA season labeled "YYYY-YY"
# starts in October of YYYY; before October we're still in the prior season.
FIRST_SEASON_START = 2022     # earliest season year to include


def current_season_start() -> int:
    d = date.today()
    return d.year if d.month >= 10 else d.year - 1


def _season_str(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def all_seasons():
    return [_season_str(y)
            for y in range(FIRST_SEASON_START, current_season_start() + 1)]


SEASONS = all_seasons()

# None = ingest all 30 teams. Set to an abbreviation (e.g. "LAL") to scope down.
INGEST_TEAM_ABBR = None

# --- Storage layout (medallion architecture) ---------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DB_PATH = DATA_DIR / "nba_load.duckdb"

# --- Rate limiting -----------------------------------------------------------
REQUEST_DELAY_SECONDS = 0.6
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
REQUEST_TIMEOUT = 30

# --- Load model params -------------------------------------------------------
ACUTE_WINDOW_DAYS = 7
CHRONIC_WINDOW_DAYS = 28
ACWR_SWEET_SPOT = (0.8, 1.3)
ACWR_DANGER = 1.5

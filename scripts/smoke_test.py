"""Quick sanity check on the gold layer via DuckDB (the analyst entrypoint).

Run after the pipeline:  python scripts/smoke_test.py
Demonstrates the window-function / aggregation SQL an analyst interview probes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import config   # noqa: E402


def main():
    con = duckdb.connect(str(config.DB_PATH))

    print("\n=== rows + date coverage ===")
    print(con.execute("""
        SELECT COUNT(*) AS player_games,
               COUNT(DISTINCT PLAYER_ID) AS players,
               MIN(GAME_DATE) AS first_game,
               MAX(GAME_DATE) AS last_game
        FROM player_load
    """).df().to_string(index=False))

    print("\n=== highest-risk player-games (danger zone) ===")
    print(con.execute("""
        SELECT PLAYER_NAME, GAME_DATE, MIN,
               ROUND(ACWR, 2) AS acwr, REST_DAYS, LOAD_FLAG
        FROM player_load
        WHERE LOAD_FLAG = 'danger'
        ORDER BY ACWR DESC
        LIMIT 10
    """).df().to_string(index=False))

    print("\n=== load-flag distribution ===")
    print(con.execute("""
        SELECT LOAD_FLAG, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM player_load
        GROUP BY LOAD_FLAG
        ORDER BY n DESC
    """).df().to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()

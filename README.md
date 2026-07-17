# NBA Player Load & Performance Intelligence Platform

End-to-end sports-analytics platform that ingests NBA game data, engineers
domain-informed **player-load** features, and flags when a player is heading
toward a fatigue-driven performance drop-off or elevated injury risk.

Built to demonstrate four role capabilities in one codebase: **Data Engineer**
(the pipeline), **Data Scientist** (the load model), **Data Analyst** (the
dashboard + SQL), and **AI Engineer** (the serving + NL query layer).

## Core question
> How does accumulated physical load affect performance and injury risk over a
> season — and can we predict a drop-off before it happens?

## Architecture

```
  nba_api (stats.nba.com)
          │  rate-limited + retry/backoff
          ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │   BRONZE     │──▶│   SILVER     │──▶│    GOLD      │
   │ raw pulls    │   │ clean/dedupe │   │ ACWR + load  │
   │ (parquet)    │   │ typed        │   │ features     │
   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                 ▼
                                          ┌──────────────┐
                                          │   DuckDB     │  ◀── SQL / dashboard
                                          │  warehouse   │
                                          └──────────────┘
```

The headline feature is the **Acute:Chronic Workload Ratio (ACWR)** — a real
sports-science metric comparing short-term load (7-day) to a rolling baseline
(28-day). It's computed on a *daily* calendar per player so windows span days,
not games.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest one team end-to-end (edit INGEST_TEAM_ABBR in config.py to scale up)
python -m src.pipeline

# Inspect the gold layer
python scripts/smoke_test.py
```

## Layout

```
config.py            # all knobs: seasons, team, rate limits, load params
src/
  ingest/
    client.py        # rate-limited, retrying nba_api wrapper
    game_logs.py     # bronze: player game logs (minutes = load proxy)
  transform/
    silver.py        # clean, type, dedupe
    gold.py          # ACWR + rest-day feature engineering
  storage/
    duckdb_io.py     # parquet IO + gold -> DuckDB loader
  pipeline.py        # bronze -> silver -> gold -> duckdb orchestration
scripts/
  smoke_test.py      # SQL sanity checks on the gold layer
```

## Roadmap

- [x] **Phase 1 — Pipeline (Data Engineer).** Idempotent bronze/silver/gold
  ingestion with rate-limit handling; DuckDB warehouse.
- [ ] **Phase 2 — Orchestration.** Wrap the pipeline in Dagster with a nightly
  schedule and asset checks.
- [ ] **Phase 3 — Model (Data Scientist).** Predict next-N-game drop-off /
  injury risk. Baseline (logistic) vs. XGBoost, **time-based** split (no
  leakage), SHAP explainability.
- [ ] **Phase 4 — Dashboard (Data Analyst).** Streamlit app: load trends, rest
  patterns, flagged high-risk windows.
- [ ] **Phase 5 — Serving (AI Engineer).** FastAPI risk endpoint + Dockerfile +
  a natural-language query agent that answers "who's at risk and why?" with
  citations back to the gold tables.

## Notes / gotchas
- `stats.nba.com` throttles hard — the retry/backoff in `client.py` is load-
  bearing, not optional.
- Tracking data (distance, avg speed) only exists from 2013-14 onward; minutes
  is used as the universal load proxy and tracking is a Phase 3 enrichment.
- **Never** random-split games for the model — split by time to avoid leakage.

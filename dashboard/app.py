"""Player Load Intelligence — Streamlit dashboard (Data Analyst layer).

Reads the gold player_load table from DuckDB and turns it into a
stakeholder-facing view: current risk board, per-player load trends, and the
league-wide flag distribution.

Run from the project root:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

FLAG_COLORS = {
    "insufficient_history": "#898781",
    "undertrained": "#2a78d6",
    "sweet_spot": "#0ca30c",
    "elevated": "#fab219",
    "danger": "#d03b3b",
}
FLAG_ORDER = ["insufficient_history", "undertrained", "sweet_spot", "elevated", "danger"]

st.set_page_config(page_title="NBA Player Load Intelligence", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(config.GOLD_DIR / "player_load.parquet")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


try:
    data = load_data()
except Exception as e:
    st.error(
        "Couldn't read the gold table. Run the pipeline first: "
        "python -m src.pipeline\n\n"
        f"Details: {e}"
    )
    st.stop()

st.sidebar.header("Filters")
seasons = sorted(data["SEASON"].unique())
season = st.sidebar.selectbox("Season", ["All"] + seasons, index=len(seasons))

df = data if season == "All" else data[data["SEASON"] == season]

players = sorted(df["PLAYER_NAME"].unique())
default_player = "LeBron James" if "LeBron James" in players else players[0]
player = st.sidebar.selectbox("Player", players, index=players.index(default_player))

st.title("NBA Player Load Intelligence")
st.caption(
    "Acute:chronic workload ratio (ACWR) from minutes played. "
    "Sweet spot 0.8-1.3; elevated 1.3-1.5; danger >1.5."
)

scored = df[df["LOAD_FLAG"] != "insufficient_history"]
danger_pct = (
    100 * (scored["LOAD_FLAG"] == "danger").sum() / len(scored) if len(scored) else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Player-games", f"{len(df):,}")
c2.metric("Players", df["PLAYER_ID"].nunique())
c3.metric("Scored games", f"{len(scored):,}")
c4.metric("In danger zone", f"{danger_pct:.1f}%")

st.divider()

st.subheader("Current risk board")
st.caption("Each player's most recent game in the selected window, highest ACWR first.")

latest = (
    df.sort_values("GAME_DATE")
    .groupby("PLAYER_NAME", as_index=False)
    .last()[["PLAYER_NAME", "GAME_DATE", "MIN", "ACWR", "REST_DAYS", "LOAD_FLAG"]]
)
latest = latest[latest["LOAD_FLAG"] != "insufficient_history"]
latest = latest.sort_values("ACWR", ascending=False)
latest["GAME_DATE"] = latest["GAME_DATE"].dt.date
latest = latest.rename(columns={
    "PLAYER_NAME": "Player", "GAME_DATE": "Last game", "MIN": "Min",
    "ACWR": "ACWR", "REST_DAYS": "Rest days", "LOAD_FLAG": "Flag",
})


def _color_flag(val):
    return f"color: {FLAG_COLORS.get(val, '#000')}; font-weight: 600;"


st.dataframe(
    latest.style.format({"ACWR": "{:.2f}", "Min": "{:.0f}"})
    .map(_color_flag, subset=["Flag"]),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader(f"Load trend — {player}")

pdf = df[df["PLAYER_NAME"] == player].sort_values("GAME_DATE")

fig = go.Figure()
fig.add_hrect(y0=0.8, y1=1.3, line_width=0, fillcolor="#0ca30c", opacity=0.08)
fig.add_hrect(y0=1.5, y1=max(2.0, pdf["ACWR"].max() or 2.0),
              line_width=0, fillcolor="#d03b3b", opacity=0.08)
fig.add_trace(go.Scatter(
    x=pdf["GAME_DATE"], y=pdf["ACWR"], mode="lines+markers",
    name="ACWR", line=dict(color="#2a78d6", width=2),
    marker=dict(size=6,
                color=[FLAG_COLORS.get(f, "#898781") for f in pdf["LOAD_FLAG"]]),
    hovertemplate="%{x|%b %d, %Y}<br>ACWR %{y:.2f}<extra></extra>",
))
fig.add_hline(y=1.5, line_dash="dot", line_color="#d03b3b",
              annotation_text="danger", annotation_position="top left")
fig.update_layout(
    height=380, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="ACWR", xaxis_title=None,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

mfig = go.Figure()
mfig.add_trace(go.Bar(
    x=pdf["GAME_DATE"], y=pdf["MIN"],
    marker_color=[FLAG_COLORS.get(f, "#898781") for f in pdf["LOAD_FLAG"]],
    hovertemplate="%{x|%b %d, %Y}<br>%{y:.0f} min<extra></extra>",
))
mfig.update_layout(
    height=220, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="Minutes", xaxis_title=None,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.caption("Minutes per game (bar color = load flag)")
st.plotly_chart(mfig, use_container_width=True)

st.divider()

st.subheader("Load-flag distribution")
counts = (
    df["LOAD_FLAG"].value_counts()
    .reindex(FLAG_ORDER).fillna(0).astype(int)
)
dfig = go.Figure(go.Bar(
    x=counts.index, y=counts.values,
    marker_color=[FLAG_COLORS[f] for f in counts.index],
))
dfig.update_layout(
    height=320, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="Player-games",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(dfig, use_container_width=True)

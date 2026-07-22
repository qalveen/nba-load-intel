"""Player Load Intelligence — Streamlit dashboard.

Filters cascade: Season -> Team -> Player. Risk board is model-driven when the
scored table is present; otherwise falls back to the ACWR load flag.

Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


@st.cache_data
def load_scored():
    path = config.DATA_DIR / "features" / "scored.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path)
    s["GAME_DATE"] = pd.to_datetime(s["GAME_DATE"])
    return s


try:
    data = load_data()
except Exception as e:
    st.error(
        "Couldn't read the gold table. Run the pipeline first: "
        "python -m src.pipeline\n\n"
        f"Details: {e}"
    )
    st.stop()

scored = load_scored()

# --- Cascading filters: Season -> Team -> Player ----------------------------
st.sidebar.header("Filters")

seasons = sorted(data["SEASON"].unique())
season = st.sidebar.selectbox("Season", ["All"] + seasons, index=len(seasons))
dseason = data if season == "All" else data[data["SEASON"] == season]

teams = sorted(dseason["TEAM_ABBREVIATION"].dropna().unique())
team = st.sidebar.selectbox("Team", ["All"] + list(teams))
df = dseason if team == "All" else dseason[dseason["TEAM_ABBREVIATION"] == team]

players = sorted(df["PLAYER_NAME"].unique())
default_idx = players.index("LeBron James") if "LeBron James" in players else 0
player = st.sidebar.selectbox("Player", players, index=default_idx)

st.title("NBA Player Load Intelligence")
st.caption(
    "Acute:chronic workload ratio (ACWR) from minutes played, with a model that "
    "predicts near-term absence risk. Sweet spot 0.8-1.3; danger >1.5."
)

scored_pl = df[df["LOAD_FLAG"] != "insufficient_history"]
danger_pct = (
    100 * (scored_pl["LOAD_FLAG"] == "danger").sum() / len(scored_pl)
    if len(scored_pl) else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Player-games", f"{len(df):,}")
c2.metric("Players", df["PLAYER_ID"].nunique())
c3.metric("Scored games", f"{len(scored_pl):,}")
c4.metric("In danger zone", f"{danger_pct:.1f}%")

st.divider()

# --- Predicted risk board ---------------------------------------------------
st.subheader("Predicted risk board")


def _color_flag(val):
    return f"color: {FLAG_COLORS.get(val, '#000')}; font-weight: 600;"


if scored is not None:
    s = scored if season == "All" else scored[scored["SEASON"] == season]
    if team != "All" and "TEAM_ABBREVIATION" in s.columns:
        s = s[s["TEAM_ABBREVIATION"] == team]

    cols = ["PLAYER_NAME"]
    if "TEAM_ABBREVIATION" in s.columns:
        cols.append("TEAM_ABBREVIATION")
    cols += ["GAME_DATE", "MIN", "ACWR", "REST_DAYS", "LOAD_FLAG", "RISK_PROBA"]

    latest = (
        s.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME", as_index=False)
        .last()[cols]
        .sort_values("RISK_PROBA", ascending=False)
    )
    st.caption("Each player's most recent game, ranked by the model's predicted "
               "probability of an upcoming absence.")
    latest["GAME_DATE"] = latest["GAME_DATE"].dt.date
    latest["RISK_PROBA"] = (latest["RISK_PROBA"] * 100).round(1)
    latest = latest.rename(columns={
        "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team",
        "GAME_DATE": "Last game", "MIN": "Min", "ACWR": "ACWR",
        "REST_DAYS": "Rest days", "LOAD_FLAG": "Flag", "RISK_PROBA": "Risk %",
    })
    st.dataframe(
        latest.style.format({"ACWR": "{:.2f}", "Min": "{:.0f}", "Risk %": "{:.1f}"})
        .map(_color_flag, subset=["Flag"])
        use_container_width=True, hide_index=True,
    )
else:
    st.info("Train and score the model to enable predicted risk: "
            "`python -m src.model.train` then `python -m src.model.score`.")
    latest = (
        df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME", as_index=False)
        .last()[["PLAYER_NAME", "GAME_DATE", "MIN", "ACWR", "REST_DAYS", "LOAD_FLAG"]]
    )
    latest = latest[latest["LOAD_FLAG"] != "insufficient_history"]
    latest = latest.sort_values("ACWR", ascending=False)
    latest["GAME_DATE"] = latest["GAME_DATE"].dt.date
    st.dataframe(latest, use_container_width=True, hide_index=True)

st.divider()

# --- Player detail ----------------------------------------------------------
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
counts = df["LOAD_FLAG"].value_counts().reindex(FLAG_ORDER).fillna(0).astype(int)
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

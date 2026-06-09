import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 — ML Prediction",
    page_icon="⚽",
    layout="wide"
)

# ── WC2026 Color Palette ─────────────────────────────────────────────────────
BLACK     = "#0A0A0A"
GOLD      = "#C9A84C"
WHITE     = "#FFFFFF"
RED       = "#E61D25"
BLUE      = "#2A398D"
GREEN     = "#3CAC3B"
DARK_GRAY = "#474A4A"
LIGHT_GRAY = "#D1D4D1"

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0A0A0A; color: #FFFFFF; }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #C9A84C;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #D1D4D1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1A1A1A;
        border: 1px solid #C9A84C;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-team {
        font-size: 1.1rem;
        font-weight: 700;
        color: #C9A84C;
    }
    .metric-prob {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #D1D4D1;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #C9A84C;
        border-bottom: 2px solid #C9A84C;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .insight-card {
        background-color: #1A1A1A;
        border-left: 4px solid #C9A84C;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Data ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/df_dashboard_data.csv"))
    with open(os.path.join(BASE_DIR, "data/processed/dashboard_insights.json")) as f:
        insights = json.load(f)
    return df, insights

df, insights = load_data()

# Banner
st.image(os.path.join(BASE_DIR, "assets/banner.png"), use_container_width=True)

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">FIFA World Cup 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML Winner Prediction — XGBoost + Monte Carlo (10,000 simulations)</div>', unsafe_allow_html=True)

# Top 3 metric cards
top3 = df.head(3)
cols = st.columns(3)
medals = ["🥇", "🥈", "🥉"]
for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-team">{medals[i]} {row["team"]}</div>
            <div class="metric-prob">{row["win_pct"]}%</div>
            <div class="metric-label">Win Probability</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Win Probability Chart ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Win Probability — Top 15</div>', unsafe_allow_html=True)

top15 = df.head(15).sort_values("win_pct", ascending=True)

fig_bar = px.bar(
    top15,
    x="win_pct",
    y="team",
    orientation="h",
    text="win_pct",
    color="win_pct",
    color_continuous_scale=[[0, DARK_GRAY], [0.5, BLUE], [1, GOLD]],
)
fig_bar.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)
fig_bar.update_layout(
    plot_bgcolor=BLACK,
    paper_bgcolor=BLACK,
    font_color=WHITE,
    xaxis_title="Win Probability (%)",
    yaxis_title="",
    coloraxis_showscale=False,
    height=500,
    margin=dict(l=20, r=60, t=20, b=20),
)
fig_bar.update_xaxes(gridcolor=DARK_GRAY)
fig_bar.update_yaxes(gridcolor=DARK_GRAY)

st.plotly_chart(fig_bar, use_container_width=True)

# ── Stage-by-Stage Table ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Stage-by-Stage Probabilities — All 48 Teams</div>', unsafe_allow_html=True)

df_table = df[[
    "team", "win_pct", "final_pct", "semi_final_pct", "quarter_final_pct"
]].copy()
df_table.columns = ["Team", "Win %", "Final %", "Semi-Final %", "Quarter-Final %"]

st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Win %":           st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.1f%%"),
        "Final %":         st.column_config.ProgressColumn("Final %", min_value=0, max_value=100, format="%.1f%%"),
        "Semi-Final %":    st.column_config.ProgressColumn("Semi-Final %", min_value=0, max_value=100, format="%.1f%%"),
        "Quarter-Final %": st.column_config.ProgressColumn("Quarter-Final %", min_value=0, max_value=100, format="%.1f%%"),
    }
)

# ── Team Explorer ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Team Explorer</div>', unsafe_allow_html=True)

selected_team = st.selectbox("Select a team", df["team"].tolist())
team_row = df[df["team"] == selected_team].iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Win Tournament", f"{team_row['win_pct']}%")
with col2:
    st.metric("Reach Final", f"{team_row['final_pct']}%")
with col3:
    st.metric("Reach Semi-Final", f"{team_row['semi_final_pct']}%")
with col4:
    st.metric("Reach Quarter-Final", f"{team_row['quarter_final_pct']}%")

# Radar chart for selected team
categories = ["Win", "Final", "Semi-Final", "Quarter-Final"]
values = [
    team_row["win_pct"],
    team_row["final_pct"],
    team_row["semi_final_pct"],
    team_row["quarter_final_pct"],
]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=values + [values[0]],
    theta=categories + [categories[0]],
    fill="toself",
    fillcolor="rgba(201, 168, 76, 0.2)",
    line=dict(color=GOLD, width=2),
    name=selected_team,
))
fig_radar.update_layout(
    polar=dict(
        bgcolor=BLACK,
        radialaxis=dict(visible=True, gridcolor=DARK_GRAY, color=LIGHT_GRAY),
        angularaxis=dict(gridcolor=DARK_GRAY, color=WHITE),
    ),
    paper_bgcolor=BLACK,
    font_color=WHITE,
    showlegend=False,
    height=400,
    margin=dict(l=40, r=40, t=40, b=40),
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Monte Carlo Insights ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Monte Carlo Insights</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**🌟 Dark Horses**")
    for t in insights["dark_horses"]:
        prob = round(t["semi_final_prob"] * 100, 1)
        st.markdown(f"""
        <div class="insight-card">
            <strong style="color:{GOLD}">{t["team"]}</strong><br>
            <span style="color:{LIGHT_GRAY}">{prob}% chance of reaching Semi-Finals</span>
        </div>
        """, unsafe_allow_html=True)

with col_b:
    st.markdown("**⚡ Biggest Upsets**")
    for t in insights["biggest_upsets"]:
        prob = round(t["quarter_final_prob"] * 100, 1)
        st.markdown(f"""
        <div class="insight-card">
            <strong style="color:{RED}">{t["team"]}</strong><br>
            <span style="color:{LIGHT_GRAY}">{prob}% chance of reaching Quarter-Finals</span>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#474A4A; font-size:0.8rem;">' +
    "Built with XGBoost + Monte Carlo simulation · 10,000 tournament simulations · " +
    "Training data: 418 historical WC matches between 2026 qualified nations" +
    "</div>",
    unsafe_allow_html=True
)

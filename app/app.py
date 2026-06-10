import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 — ML Prediction",
    page_icon="⚽",
    layout="wide"
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models"
ASSETS    = BASE_DIR / "assets"

# ── Mascot-Inspired Palette ──────────────────────────────────────────────────
BG         = "#0A0A12"
SURFACE    = "#141420"
PRIMARY    = "#C9A84C"
SECONDARY  = "#C0392B"
ACCENT     = "#1B5E20"
TEXT       = "#E8EAF6"
TEXT_MUTED = "#B0BEC5"
SUCCESS    = "#2E7D32"
WARNING    = "#D4A017"
DANGER     = "#C0392B"

# ── Flag Emojis ──────────────────────────────────────────────────────────────
FLAGS = {
    "Argentina":            "🇦🇷",
    "Algeria":              "🇩🇿",
    "Australia":            "🇦🇺",
    "Austria":              "🇦🇹",
    "Belgium":              "🇧🇪",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil":               "🇧🇷",
    "Cape Verde":           "🇨🇻",
    "Canada":               "🇨🇦",
    "Colombia":             "🇨🇴",
    "Croatia":              "🇭🇷",
    "Czechia":              "🇨🇿",
    "Ivory Coast":          "🇨🇮",
    "Curaçao":              "🇨🇼",
    "DR Congo":             "🇨🇩",
    "Ecuador":              "🇪🇨",
    "Egypt":                "🇪🇬",
    "England":              "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France":               "🇫🇷",
    "Germany":              "🇩🇪",
    "Ghana":                "🇬🇭",
    "Haiti":                "🇭🇹",
    "Iran":                 "🇮🇷",
    "Iraq":                 "🇮🇶",
    "Japan":                "🇯🇵",
    "Jordan":               "🇯🇴",
    "Mexico":               "🇲🇽",
    "Morocco":              "🇲🇦",
    "Netherlands":          "🇳🇱",
    "New Zealand":          "🇳🇿",
    "Norway":               "🇳🇴",
    "Panama":               "🇵🇦",
    "Paraguay":             "🇵🇾",
    "Portugal":             "🇵🇹",
    "Qatar":                "🇶🇦",
    "Saudi Arabia":         "🇸🇦",
    "Scotland":             "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Senegal":              "🇸🇳",
    "South Africa":         "🇿🇦",
    "South Korea":          "🇰🇷",
    "Spain":                "🇪🇸",
    "Sweden":               "🇸🇪",
    "Switzerland":          "🇨🇭",
    "Tunisia":              "🇹🇳",
    "Turkey":               "🇹🇷",
    "Uruguay":              "🇺🇾",
    "United States":        "🇺🇸",
    "Uzbekistan":           "🇺🇿",
}

def flag(team):
    return FLAGS.get(team, "🏳️")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{ background-color: #0D0D1A; }}
    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {PRIMARY};
        text-align: center;
        margin-bottom: 0.2rem;
    }}
    .subtitle {{
        font-size: 1.1rem;
        color: {TEXT_MUTED};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {SURFACE};
        border: 1px solid {PRIMARY};
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }}
    .metric-team {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {PRIMARY};
    }}
    .metric-prob {{
        font-size: 2rem;
        font-weight: 800;
        color: {TEXT};
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
    }}
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {PRIMARY};
        border-bottom: 2px solid {PRIMARY};
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }}
    .insight-card {{
        background-color: {SURFACE};
        border-left: 4px solid {PRIMARY};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }}
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED / "df_dashboard_data.csv")
    with open(PROCESSED / "dashboard_insights.json") as f:
        insights = json.load(f)
    return df, insights

@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "xgb_match_predictor.pkl")
    with open(MODELS / "feature_list.json") as f:
        feature_list = json.load(f)
    return model, feature_list

df, insights = load_data()
model, feature_list = load_model()

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.image(str(ASSETS / "banner.png"), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    view = st.radio(
        "Navigation",
        ["🏆 Hero", "🗂 Tournament Bracket", "🔍 Team Profile",
         "🎲 Monte Carlo Insights", "📊 Feature Importance"],
        label_visibility="collapsed"
    )

# ════════════════════════════════════════════════════════════════════════════
# VIEW 1 — HERO
# ════════════════════════════════════════════════════════════════════════════
if view == "🏆 Hero":
    st.markdown('<div class="main-title">⚽ FIFA World Cup 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">ML Winner Prediction — XGBoost + Monte Carlo (10,000 simulations)</div>', unsafe_allow_html=True)

    top3 = df.head(3)
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-team">{medals[i]} {flag(row['team'])} {row['team']}</div>
                <div class="metric-prob">{row['win_pct']}%</div>
                <div class="metric-label">Win Probability</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Win Probability — Top 15</div>', unsafe_allow_html=True)

    top15 = df.head(15).copy()
    top15["label"] = top15["team"].apply(lambda t: f"{flag(t)} {t}")
    top15 = top15.sort_values("win_pct", ascending=True)

    fig_bar = px.bar(
        top15, x="win_pct", y="label", orientation="h",
        text="win_pct",
        color="win_pct",
        color_continuous_scale=[[0, SURFACE], [0.5, WARNING], [1, PRIMARY]],
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
        xaxis_title="Win Probability (%)", yaxis_title="",
        coloraxis_showscale=False, height=500,
        margin=dict(l=20, r=60, t=20, b=20),
    )
    fig_bar.update_xaxes(gridcolor=SURFACE)
    fig_bar.update_yaxes(gridcolor=SURFACE)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">Stage-by-Stage Probabilities — All 48 Teams</div>', unsafe_allow_html=True)
    df_table = df[["team", "win_pct", "final_pct", "semi_final_pct", "quarter_final_pct"]].copy()
    df_table["team"] = df_table["team"].apply(lambda t: f"{flag(t)} {t}")
    df_table.columns = ["Team", "Win %", "Final %", "Semi-Final %", "Quarter-Final %"]
    st.dataframe(
        df_table, use_container_width=True, hide_index=True,
        column_config={
            "Win %":           st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.1f%%"),
            "Final %":         st.column_config.ProgressColumn("Final %", min_value=0, max_value=100, format="%.1f%%"),
            "Semi-Final %":    st.column_config.ProgressColumn("Semi-Final %", min_value=0, max_value=100, format="%.1f%%"),
            "Quarter-Final %": st.column_config.ProgressColumn("Quarter-Final %", min_value=0, max_value=100, format="%.1f%%"),
        }
    )

# ════════════════════════════════════════════════════════════════════════════
# VIEW 2 — TOURNAMENT BRACKET
# ════════════════════════════════════════════════════════════════════════════
elif view == "🗂 Tournament Bracket":
    st.markdown('<div class="section-title">Tournament Bracket — Predicted Path</div>', unsafe_allow_html=True)
    st.caption("Most likely bracket based on 10,000 Monte Carlo simulations. Color: gold = high probability, red = low.")

    # Teams per stage sorted by probability
    r32_teams = df.nlargest(32, "quarter_final_prob")["team"].tolist()
    r16_teams = df.nlargest(16, "quarter_final_prob")["team"].tolist()
    qf_teams  = df.nlargest(8,  "quarter_final_prob")["team"].tolist()
    sf_teams  = df.nlargest(4,  "semi_final_prob")["team"].tolist()
    f_teams   = df.nlargest(2,  "final_prob")["team"].tolist()
    winner    = df.nlargest(1,  "win_prob")["team"].iloc[0]

    stages_left  = [r32_teams[:16], r16_teams[:8], qf_teams[:4], sf_teams[:2]]
    stages_right = [r32_teams[16:], r16_teams[8:], qf_teams[4:], sf_teams[2:]]
    stage_names  = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
    prob_cols    = ["quarter_final_pct", "quarter_final_pct", "quarter_final_pct", "semi_final_pct"]

    fig = go.Figure()

    box_w = 0.13
    box_h = 0.045
    center_x = 0.5

    def get_color(prob):
        t = min(prob / 40, 1.0)
        r = int(192 + (201 - 192) * t)
        g = int(57  + (168 - 57)  * t)
        b = int(30  + (76  - 30)  * t)
        return f"rgba({r},{g},{b},0.85)"

    def draw_team(fig, x, y, team, prob, align="left"):
        color = get_color(prob)
        x0 = x if align == "left" else x - box_w
        x1 = x + box_w if align == "left" else x
        fig.add_shape(type="rect", x0=x0, y0=y-box_h/2, x1=x1, y1=y+box_h/2,
                      fillcolor=color, line=dict(color=PRIMARY, width=1))
        fig.add_annotation(
            x=(x0+x1)/2, y=y + 0.008,
            text=f"<b>{flag(team)} {team}</b>",
            showarrow=False, font=dict(color=TEXT, size=9), xanchor="center"
        )
        fig.add_annotation(
            x=(x0+x1)/2, y=y - 0.015,
            text=f"{prob:.1f}%",
            showarrow=False, font=dict(color=TEXT_MUTED, size=8), xanchor="center"
        )

    # Left side columns: x positions moving right toward center
    left_x  = [0.01, 0.16, 0.29, 0.40]
    right_x = [0.86, 0.71, 0.58, 0.47]

    for s_idx in range(4):
        l_teams = stages_left[s_idx]
        r_teams = stages_right[s_idx]
        prob_col = prob_cols[s_idx]
        n = len(l_teams)
        y_pos = np.linspace(0.93, 0.07, n)

        for t_idx, team in enumerate(l_teams):
            prob = df[df["team"] == team][prob_col].values[0]
            draw_team(fig, left_x[s_idx], y_pos[t_idx], team, prob, align="left")

        for t_idx, team in enumerate(r_teams):
            prob = df[df["team"] == team][prob_col].values[0]
            draw_team(fig, right_x[s_idx], y_pos[t_idx], team, prob, align="right")

        # Stage labels
        fig.add_annotation(
            x=left_x[s_idx] + box_w/2, y=0.98,
            text=f"<b>{stage_names[s_idx]}</b>",
            showarrow=False, font=dict(color=PRIMARY, size=10), xanchor="center"
        )
        fig.add_annotation(
            x=right_x[s_idx] - box_w/2, y=0.98,
            text=f"<b>{stage_names[s_idx]}</b>",
            showarrow=False, font=dict(color=PRIMARY, size=10), xanchor="center"
        )

    # Finalists — center
    final_y = [0.58, 0.42]
    for i, team in enumerate(f_teams):
        prob = df[df["team"] == team]["final_pct"].values[0]
        draw_team(fig, center_x - box_w/2, final_y[i], team, prob, align="left")

    fig.add_annotation(x=center_x, y=0.98, text="<b>Final</b>",
                       showarrow=False, font=dict(color=PRIMARY, size=11), xanchor="center")

    # Winner — center top
    w_prob = df[df["team"] == winner]["win_pct"].values[0]
    fig.add_shape(type="rect",
                  x0=center_x - box_w/2 - 0.01, y0=0.68,
                  x1=center_x + box_w/2 + 0.01, y1=0.78,
                  fillcolor=PRIMARY, line=dict(color=WARNING, width=2))
    fig.add_annotation(x=center_x, y=0.745,
                       text=f"🏆 <b>{flag(winner)} {winner}</b>",
                       showarrow=False, font=dict(color=BG, size=12), xanchor="center")
    fig.add_annotation(x=center_x, y=0.705,
                       text=f"{w_prob:.1f}% win probability",
                       showarrow=False, font=dict(color=BG, size=9), xanchor="center")

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        height=900,
        margin=dict(l=5, r=5, t=20, b=5),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1.02]),
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 3 — TEAM PROFILE
# ════════════════════════════════════════════════════════════════════════════
elif view == "🔍 Team Profile":
    st.markdown('<div class="section-title">Team Profile</div>', unsafe_allow_html=True)

    team_options = [f"{flag(t)} {t}" for t in df["team"].tolist()]
    selected_label = st.selectbox("Select a team", team_options)
    selected_team  = selected_label.split(" ", 1)[1] if " " in selected_label else selected_label
    # Handle multi-word flag emojis
    selected_team = df["team"][df["team"].apply(lambda t: f"{flag(t)} {t}") == selected_label].values[0]
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

    categories = ["Win", "Final", "Semi-Final", "Quarter-Final"]
    values = [team_row["win_pct"], team_row["final_pct"],
              team_row["semi_final_pct"], team_row["quarter_final_pct"]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=f"rgba(201,168,76,0.2)",
        line=dict(color=PRIMARY, width=2),
        name=selected_team,
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, gridcolor=SURFACE, color=TEXT_MUTED),
            angularaxis=dict(gridcolor=SURFACE, color=TEXT),
        ),
        paper_bgcolor=BG, font_color=TEXT,
        showlegend=False, height=400,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    fig_stages = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=[PRIMARY, WARNING, WARNING, WARNING],
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig_stages.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
        yaxis_title="Probability (%)", xaxis_title="",
        height=350, margin=dict(l=20, r=20, t=20, b=20),
    )
    fig_stages.update_yaxes(gridcolor=SURFACE)
    st.plotly_chart(fig_stages, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 4 — MONTE CARLO INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif view == "🎲 Monte Carlo Insights":
    st.markdown('<div class="section-title">Monte Carlo Insights</div>', unsafe_allow_html=True)

    df_plot = df.sort_values("win_pct", ascending=False).copy()
    df_plot["label"] = df_plot["team"].apply(lambda t: f"{flag(t)} {t}")

    fig_dist = px.bar(
        df_plot, x="label", y="win_pct",
        color="win_pct",
        color_continuous_scale=[[0, SURFACE], [0.5, WARNING], [1, PRIMARY]],
    )
    fig_dist.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
        xaxis_title="", yaxis_title="Win Probability (%)",
        coloraxis_showscale=False, height=400,
        margin=dict(l=20, r=20, t=20, b=100),
    )
    fig_dist.update_xaxes(tickangle=45, gridcolor=SURFACE)
    fig_dist.update_yaxes(gridcolor=SURFACE)
    st.plotly_chart(fig_dist, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🌟 Dark Horses**")
        for t in insights["dark_horses"]:
            prob = round(t["semi_final_prob"] * 100, 1)
            st.markdown(f"""
            <div class="insight-card">
                <strong style="color:{PRIMARY}">{flag(t['team'])} {t['team']}</strong><br>
                <span style="color:{TEXT_MUTED}">{prob}% chance of reaching Semi-Finals</span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("**⚡ Biggest Upsets**")
        for t in insights["biggest_upsets"]:
            prob = round(t["quarter_final_prob"] * 100, 1)
            st.markdown(f"""
            <div class="insight-card">
                <strong style="color:{SECONDARY}">{flag(t['team'])} {t['team']}</strong><br>
                <span style="color:{TEXT_MUTED}">{prob}% chance of reaching Quarter-Finals</span>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 5 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
elif view == "📊 Feature Importance":
    st.markdown('<div class="section-title">XGBoost Feature Importance</div>', unsafe_allow_html=True)
    st.caption("Features ranked by F-score — how often each feature was used to split nodes in the XGBoost trees.")

    importances = pd.Series(
        model.feature_importances_,
        index=feature_list
    ).sort_values(ascending=True)

    fig_imp = px.bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        color=importances.values,
        color_continuous_scale=[[0, SURFACE], [0.5, WARNING], [1, PRIMARY]],
    )
    fig_imp.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
        xaxis_title="F-score", yaxis_title="",
        coloraxis_showscale=False, height=500,
        margin=dict(l=20, r=40, t=20, b=20),
    )
    fig_imp.update_xaxes(gridcolor=SURFACE)
    fig_imp.update_yaxes(gridcolor=SURFACE)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card">
        <strong style="color:{PRIMARY}">Model Performance</strong><br>
        <span style="color:{TEXT_MUTED}">
        Binary XGBoost classifier · Target: Home Win vs No Home Win ·
        Accuracy: 71% · Training set: 418 historical WC matches between 2026 qualified nations
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center; color:{TEXT_MUTED}; font-size:0.8rem;">'
    "Built with XGBoost + Monte Carlo simulation · 10,000 tournament simulations · "
    "Training data: 418 historical WC matches between 2026 qualified nations"
    "</div>",
    unsafe_allow_html=True
)
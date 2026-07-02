import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(page_title="WC 2026 — ML Prediction", page_icon="⚽", layout="wide")

BASE_DIR  = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models"
ASSETS    = BASE_DIR / "assets"

BG         = "#13131F"
SURFACE    = "#1E1E30"
SURFACE2   = "#252540"
PRIMARY    = "#C9A84C"
SECONDARY  = "#C0392B"
TEXT       = "#F0F0FA"
TEXT_MUTED = "#9090B0"
WARNING    = "#D4A017"
BORDER     = "#2E2E48"

FLAGS = {
    "Argentina":              "\U0001F1E6\U0001F1F7",
    "Algeria":                "\U0001F1E9\U0001F1FF",
    "Australia":              "\U0001F1E6\U0001F1FA",
    "Austria":                "\U0001F1E6\U0001F1F9",
    "Belgium":                "\U0001F1E7\U0001F1EA",
    "Bosnia and Herzegovina": "\U0001F1E7\U0001F1E6",
    "Brazil":                 "\U0001F1E7\U0001F1F7",
    "Cape Verde":             "\U0001F1E8\U0001F1FB",
    "Canada":                 "\U0001F1E8\U0001F1E6",
    "Colombia":               "\U0001F1E8\U0001F1F4",
    "Croatia":                "\U0001F1ED\U0001F1F7",
    "Czechia":                "\U0001F1E8\U0001F1FF",
    "Ivory Coast":            "\U0001F1E8\U0001F1EE",
    "Cura\u00e7ao":           "\U0001F1E8\U0001F1FC",
    "DR Congo":               "\U0001F1E8\U0001F1E9",
    "Ecuador":                "\U0001F1EA\U0001F1E8",
    "Egypt":                  "\U0001F1EA\U0001F1EC",
    "England":                "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "France":                 "\U0001F1EB\U0001F1F7",
    "Germany":                "\U0001F1E9\U0001F1EA",
    "Ghana":                  "\U0001F1EC\U0001F1ED",
    "Haiti":                  "\U0001F1ED\U0001F1F9",
    "Iran":                   "\U0001F1EE\U0001F1F7",
    "Iraq":                   "\U0001F1EE\U0001F1F6",
    "Japan":                  "\U0001F1EF\U0001F1F5",
    "Jordan":                 "\U0001F1EF\U0001F1F4",
    "Mexico":                 "\U0001F1F2\U0001F1FD",
    "Morocco":                "\U0001F1F2\U0001F1E6",
    "Netherlands":            "\U0001F1F3\U0001F1F1",
    "New Zealand":            "\U0001F1F3\U0001F1FF",
    "Norway":                 "\U0001F1F3\U0001F1F4",
    "Panama":                 "\U0001F1F5\U0001F1E6",
    "Paraguay":               "\U0001F1F5\U0001F1FE",
    "Portugal":               "\U0001F1F5\U0001F1F9",
    "Qatar":                  "\U0001F1F6\U0001F1E6",
    "Saudi Arabia":           "\U0001F1F8\U0001F1E6",
    "Scotland":               "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "Senegal":                "\U0001F1F8\U0001F1F3",
    "South Africa":           "\U0001F1FF\U0001F1E6",
    "South Korea":            "\U0001F1F0\U0001F1F7",
    "Spain":                  "\U0001F1EA\U0001F1F8",
    "Sweden":                 "\U0001F1F8\U0001F1EA",
    "Switzerland":            "\U0001F1E8\U0001F1ED",
    "Tunisia":                "\U0001F1F9\U0001F1F3",
    "Turkey":                 "\U0001F1F9\U0001F1F7",
    "Uruguay":                "\U0001F1FA\U0001F1FE",
    "United States":          "\U0001F1FA\U0001F1F8",
    "Uzbekistan":             "\U0001F1FA\U0001F1FF",
}
def flag(team): return FLAGS.get(team, "\U0001F3F3")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
    .stApp {{ background-color:{BG}; color:{TEXT}; font-family:'Noto Color Emoji',Arial,sans-serif; }}
    [data-testid="stSidebar"] {{ background-color:#0F0F1C; border-right:1px solid {BORDER}; }}
    .main-title {{ font-size:2.8rem; font-weight:800; color:{PRIMARY}; text-align:center; margin-bottom:0.2rem; }}
    .subtitle {{ font-size:1.1rem; color:{TEXT_MUTED}; text-align:center; margin-bottom:2rem; }}
    .metric-card {{ background-color:{SURFACE}; border:1px solid {BORDER}; border-top:3px solid {PRIMARY}; border-radius:12px; padding:1.4rem; text-align:center; }}
    .metric-team {{ font-size:1.05rem; font-weight:700; color:{PRIMARY}; margin-bottom:0.4rem; }}
    .metric-prob {{ font-size:2.2rem; font-weight:800; color:{TEXT}; }}
    .metric-label {{ font-size:0.78rem; color:{TEXT_MUTED}; margin-top:0.2rem; }}
    .section-title {{ font-size:1.4rem; font-weight:700; color:{PRIMARY}; border-bottom:1px solid {BORDER}; padding-bottom:0.4rem; margin-bottom:1.2rem; margin-top:1.5rem; }}
    .insight-card {{ background-color:{SURFACE}; border:1px solid {BORDER}; border-left:4px solid {PRIMARY}; border-radius:8px; padding:1rem; margin-bottom:0.8rem; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED / "df_dashboard_data.csv")
    with open(PROCESSED / "dashboard_insights.json") as f:
        insights = json.load(f)
    return df, insights

@st.cache_data
def load_bracket_positions():
    return pd.read_csv(PROCESSED / "df_bracket_positions.csv")

@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "xgb_match_predictor.pkl")
    with open(MODELS / "feature_list.json") as f:
        feature_list = json.load(f)
    return model, feature_list

df, insights = load_data()
model, feature_list = load_model()
bp = load_bracket_positions()

# ── Group positions lookup ────────────────────────────────────────────────────
gp = {}
for _, row in bp.iterrows():
    g, p, t = row['group'], int(row['position']), row['team']
    if g not in gp: gp[g] = {}
    gp[g][p] = t

thirds = bp[bp['position']==3].sort_values('probability', ascending=False)
thirds_list = [(row['team'], row['group']) for _, row in thirds.iterrows()]

third_slot_groups = {
    'ABCDF': set('ABCDF'), 'CDFGH': set('CDFGH'),
    'CEFHI': set('CEFHI'), 'EHIJK': set('EHIJK'),
    'AEHIJ': set('AEHIJ'), 'BEFIJ': set('BEFIJ'),
    'EFGIJ': set('EFGIJ'), 'DEIJL': set('DEIJL'),
}

assigned_thirds = {
    'ABCDF': 'Paraguay',
    'CDFGH': 'Sweden',
    'CEFHI': 'Ecuador',
    'EHIJK': 'DR Congo',
    'AEHIJ': 'Senegal',
    'BEFIJ': 'Bosnia and Herzegovina',
    'EFGIJ': 'Algeria',
    'DEIJL': 'Ghana',
}

def resolve_slot(slot):
    pos = int(slot[0])
    groups = slot[1:]
    if pos in [1, 2]:
        return gp.get(groups, {}).get(pos, "TBD")
    return assigned_thirds.get(groups, "TBD")

R32_BRACKET = {
    73:('2A','2B'), 74:('1C','2F'), 75:('1E','3ABCDF'),
    76:('1F','2C'), 77:('2E','2I'), 78:('1I','3CDFGH'),
    79:('1A','3CEFHI'), 80:('1L','3EHIJK'), 81:('1G','3AEHIJ'),
    82:('1D','3BEFIJ'), 83:('1H','2J'), 84:('2K','2L'),
    85:('1B','3EFGIJ'), 86:('2D','2G'), 87:('1J','2H'),
    88:('1K','3DEIJL'),
}
R16_BRACKET = {89:(73,75), 90:(74,77), 91:(76,78), 92:(79,80),
               93:(83,84), 94:(81,82), 95:(86,88), 96:(85,87)}
QF_BRACKET  = {97:(89,90), 98:(93,94), 99:(91,92), 100:(95,96)}
SF_BRACKET  = {101:(97,98), 102:(99,100)}

win_prob = dict(zip(df['team'], df['win_prob']))
def likely_winner(t1, t2):
    return t1 if win_prob.get(t1,0.5) >= win_prob.get(t2,0.5) else t2

r32_teams = {mid: (resolve_slot(h), resolve_slot(a)) for mid,(h,a) in R32_BRACKET.items()}
match_winners = {mid: likely_winner(h,a) for mid,(h,a) in r32_teams.items()}
r16_matches = {}
for mid,(r1,r2) in R16_BRACKET.items():
    h,a = match_winners[r1], match_winners[r2]
    r16_matches[mid] = (h,a)
    match_winners[mid] = likely_winner(h,a)
qf_matches = {}
for mid,(r1,r2) in QF_BRACKET.items():
    h,a = match_winners[r1], match_winners[r2]
    qf_matches[mid] = (h,a)
    match_winners[mid] = likely_winner(h,a)
sf_matches = {}
for mid,(q1,q2) in SF_BRACKET.items():
    h,a = match_winners[q1], match_winners[q2]
    sf_matches[mid] = (h,a)
    match_winners[mid] = likely_winner(h,a)
final_teams = (match_winners[101], match_winners[102])
winner = likely_winner(*final_teams)

# FIX: get_pct reads win_pct/final_pct etc which are already percentages (0-100)
# do NOT multiply by 100
def get_pct(team, col):
    v = df[df['team']==team][col].values
    return round(float(v[0]), 1) if len(v) else 0.0

# ── Load results for live indicator ──────────────────────────────────────────
@st.cache_data
def load_results():
    results_path = PROCESSED / "results.csv"
    if results_path.exists():
        df_r = pd.read_csv(results_path)
        return df_r
    return pd.DataFrame()

df_results = load_results()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(str(ASSETS / "banner.png"), use_container_width=True)
    except Exception:
        st.markdown('<div style="text-align:center;font-size:2rem;padding:1rem;">⚽</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    view = st.radio("Navigation",
        ["🏆 Hero","🗂 Tournament Bracket","🔍 Team Profile","🎲 Monte Carlo Insights","📊 Feature Importance","🧪 Model Validation"],
        label_visibility="collapsed")

    # Live update indicator
    st.markdown("<br>", unsafe_allow_html=True)
    if not df_results.empty:
        locked   = df_results[df_results['home_score'].notna() & (df_results['home_score'] != '')]
        pending  = df_results[df_results['home_score'].isna() | (df_results['home_score'] == '')]
        n_locked  = len(locked)
        n_pending = len(pending)
        st.markdown(f"""
        <div style="background:#1E1E30;border:1px solid #2E2E48;border-left:3px solid #C9A84C;
                    border-radius:6px;padding:10px 12px;font-size:11px;">
            <div style="color:#C9A84C;font-weight:bold;margin-bottom:6px;">🔴 Live Tournament</div>
            <div style="color:#F0F0FA;">✅ {n_locked} matches locked</div>
            <div style="color:#9090B0;">⏳ {n_pending} matches remaining</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#1E1E30;border:1px solid #2E2E48;border-left:3px solid #2E2E48;
                    border-radius:6px;padding:10px 12px;font-size:11px;">
            <div style="color:#9090B0;">⏳ Tournament not started</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════
# VIEW 1 — HERO
# ════════════════════════════════
if view == "🏆 Hero":
    st.markdown('<div class="main-title">⚽ FIFA World Cup 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">ML Winner Prediction — XGBoost + Monte Carlo (10,000 simulations)</div>', unsafe_allow_html=True)
    top3 = df.head(3)
    cols = st.columns(3)
    medals = ["🥇","🥈","🥉"]
    for i,(col,(_,row)) in enumerate(zip(cols, top3.iterrows())):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-team">{medals[i]} {flag(row['team'])} {row['team']}</div>
                <div class="metric-prob">{row['win_pct']}%</div>
                <div class="metric-label">Win Probability</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Win Probability — Top 15</div>', unsafe_allow_html=True)
    top15 = df.head(15).copy()
    top15["label"] = top15["team"].apply(lambda t: f"{flag(t)} {t}")
    top15 = top15.sort_values("win_pct", ascending=True)
    fig_bar = px.bar(top15, x="win_pct", y="label", orientation="h", text="win_pct",
                     color="win_pct", color_continuous_scale=[[0,SURFACE2],[0.5,WARNING],[1,PRIMARY]])
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                          xaxis_title="Win Probability (%)", yaxis_title="",
                          coloraxis_showscale=False, height=500, margin=dict(l=20,r=60,t=20,b=20))
    fig_bar.update_xaxes(gridcolor=BORDER); fig_bar.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('<div class="section-title">Stage-by-Stage Probabilities — All 48 Teams</div>', unsafe_allow_html=True)
    df_table = df[["team","win_pct","final_pct","semi_final_pct","quarter_final_pct"]].copy()
    df_table["team"] = df_table["team"].apply(lambda t: f"{flag(t)} {t}")
    df_table.columns = ["Team","Win %","Final %","Semi-Final %","Quarter-Final %"]
    st.dataframe(df_table, use_container_width=True, hide_index=True,
        column_config={
            "Win %":           st.column_config.ProgressColumn("Win %",           min_value=0, max_value=100, format="%.1f%%"),
            "Final %":         st.column_config.ProgressColumn("Final %",         min_value=0, max_value=100, format="%.1f%%"),
            "Semi-Final %":    st.column_config.ProgressColumn("Semi-Final %",    min_value=0, max_value=100, format="%.1f%%"),
            "Quarter-Final %": st.column_config.ProgressColumn("Quarter-Final %", min_value=0, max_value=100, format="%.1f%%"),
        })

# ════════════════════════════════
# VIEW 2 — TOURNAMENT BRACKET
# ════════════════════════════════
elif view == "🗂 Tournament Bracket":
    st.markdown('<div class="section-title">Tournament Bracket — Predicted Path</div>', unsafe_allow_html=True)
    st.caption("Official 2026 bracket structure. Most likely team per slot based on 10,000 Monte Carlo simulations.")

    # Box height + gap constants — used to compute vertical alignment across stages
    BH   = 46   # px per team box (including internal padding)
    GAP  = 6    # px gap between two teams in a match
    MGAP = 18   # px gap between matches in same stage

    # A match block = 2 boxes + gap between them
    MATCH_H = BH * 2 + GAP

    # R32: 8 matches per side
    # Each pair of R32 matches feeds 1 R16 match
    # R16 match center = midpoint of its 2 R32 match centers
    # QF match center  = midpoint of its 2 R16 match centers
    # SF match center  = midpoint of its 2 QF match centers

    def match_center(match_idx, stage_matches):
        """Y center of a match given its index in the stage list."""
        return match_idx * (MATCH_H + MGAP) + MATCH_H / 2

    # R32 centers (8 matches)
    r32_centers = [match_center(i, 8) for i in range(8)]
    # R16 centers: average of pairs of R32
    r16_centers = [(r32_centers[i*2] + r32_centers[i*2+1]) / 2 for i in range(4)]
    # QF centers: average of pairs of R16
    qf_centers  = [(r16_centers[i*2] + r16_centers[i*2+1]) / 2 for i in range(2)]
    # SF centers: average of pairs of QF
    sf_center   = (qf_centers[0] + qf_centers[1]) / 2

    total_h = 8 * (MATCH_H + MGAP) - MGAP  # total height of R32 column

    def px(v): return f"{v:.1f}px"

    def team_box(team, prob, winner=False):
        f = flag(team)
        name = team[:14]+"…" if len(team)>15 else team
        bg = "rgba(110,85,20,0.98)" if winner else "rgba(45,35,10,0.95)"
        border = "#C9A84C" if winner else "#3a3010"
        return (
            f'<div style="background:{bg};border:1px solid {border};border-radius:4px;'
            f'padding:5px 7px;font-size:8px;color:#F0F0FA;text-align:center;'
            f'width:126px;height:{BH}px;box-sizing:border-box;'
            f'display:flex;flex-direction:column;justify-content:center;line-height:1.3;">'
            f'{f} {name}<br>'
            f'<span style="color:#9090B0;font-size:7px;">{prob:.1f}%</span>'
            f'</div>'
        )

    def match_html(team1, prob1, team2, prob2, w):
        return (
            team_box(team1, prob1, team1==w) +
            f'<div style="height:{GAP}px"></div>' +
            team_box(team2, prob2, team2==w)
        )

    def stage_col(label, matches_data, centers, col_offset=0):
        """
        matches_data: list of (team1, prob1, team2, prob2, winner)
        centers: list of Y centers for each match
        """
        items = [f'<div style="color:#C9A84C;font-size:9px;font-weight:bold;text-align:center;'
                 f'letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">{label}</div>']
        items.append(f'<div style="position:relative;height:{px(total_h)};">')
        for i, (t1,p1,t2,p2,w) in enumerate(matches_data):
            top = centers[i] - MATCH_H / 2
            items.append(
                f'<div style="position:absolute;top:{px(top)};left:0;">'
                + match_html(t1,p1,t2,p2,w) +
                f'</div>'
            )
        items.append('</div>')
        return "".join(items)

    def sf_col(label, team1, prob1, team2, prob2, w):
        top = sf_center - MATCH_H / 2
        return (
            f'<div style="color:#C9A84C;font-size:9px;font-weight:bold;text-align:center;'
            f'letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">{label}</div>'
            f'<div style="position:relative;height:{px(total_h)};">'
            f'<div style="position:absolute;top:{px(top)};left:0;">'
            + match_html(team1,prob1,team2,prob2,w) +
            f'</div></div>'
        )

    def final_col(f0, fp0, f1, fp1, win, wp):
        fin_top  = sf_center - MATCH_H / 2
        win_top  = sf_center - BH / 2
        return (
            f'<div style="color:#C9A84C;font-size:10px;font-weight:bold;text-align:center;'
            f'letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">⚽ Final</div>'
            f'<div style="position:relative;height:{px(total_h)};min-width:148px;">'
            # finalist top
            f'<div style="position:absolute;top:{px(fin_top)};left:0;">'
            + team_box(f0, fp0, False) +
            f'</div>'
            # winner box
            f'<div style="position:absolute;top:{px(win_top - BH - 40)};left:-8px;'
            f'background:#C9A84C;border:2px solid #D4A017;border-radius:6px;'
            f'padding:8px 10px;font-size:10px;font-weight:bold;color:#13131F;'
            f'text-align:center;width:144px;box-sizing:border-box;">'
            f'🏆 {flag(win)} {win}<br>'
            f'<span style="font-size:8px;color:#3a2a05;">{wp:.1f}% win probability</span>'
            f'</div>'
            # finalist bottom
            f'<div style="position:absolute;top:{px(fin_top + BH + GAP)};left:0;">'
            + team_box(f1, fp1, False) +
            f'</div>'
            f'</div>'
        )

    # Build match data for each stage
    left_r32_order  = [73, 75, 74, 77, 83, 84, 81, 82]
    right_r32_order = [76, 78, 79, 80, 86, 88, 85, 87]
    left_r16_order  = [89, 90, 93, 94]
    right_r16_order = [91, 92, 95, 96]
    left_qf_order   = [97, 98]
    right_qf_order  = [99, 100]

    def r32_data(order):
        data = []
        for mid in order:
            h,a = r32_teams[mid]
            w = match_winners[mid]
            data.append((h, get_pct(h,'quarter_final_pct'), a, get_pct(a,'quarter_final_pct'), w))
        return data

    def r16_data(order):
        data = []
        for mid in order:
            h,a = r16_matches[mid]
            w = match_winners[mid]
            data.append((h, get_pct(h,'semi_final_pct'), a, get_pct(a,'semi_final_pct'), w))
        return data

    def qf_data(order):
        data = []
        for mid in order:
            h,a = qf_matches[mid]
            w = match_winners[mid]
            data.append((h, get_pct(h,'final_pct'), a, get_pct(a,'final_pct'), w))
        return data

    sf_L_h, sf_L_a = sf_matches[101]; sf_L_w = match_winners[101]
    sf_R_h, sf_R_a = sf_matches[102]; sf_R_w = match_winners[102]
    f0, f1 = final_teams
    fp0 = get_pct(f0,'final_pct'); fp1 = get_pct(f1,'final_pct')
    wp  = get_pct(winner,'win_pct')

    COL_W  = "134px"
    GAP_W  = "12px"

    html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
* {{ box-sizing:border-box; font-family:'Noto Color Emoji',Arial,sans-serif; }}
.bk {{ display:flex; align-items:flex-start; background:#13131F;
       padding:16px 8px; gap:0; overflow-x:auto; }}
.col {{ min-width:{COL_W}; flex-shrink:0; }}
.gap {{ width:{GAP_W}; flex-shrink:0; }}
.center-col {{ min-width:160px; flex-shrink:0; }}
</style>
<div class="bk">
  <div class="col">{stage_col("Round of 32",  r32_data(left_r32_order),  r32_centers)}</div>
  <div class="gap"></div>
  <div class="col">{stage_col("Round of 16",  r16_data(left_r16_order),  r16_centers)}</div>
  <div class="gap"></div>
  <div class="col">{stage_col("Quarter-Finals", qf_data(left_qf_order),  qf_centers)}</div>
  <div class="gap"></div>
  <div class="col">{sf_col("Semi-Finals", sf_L_h, get_pct(sf_L_h,'semi_final_pct'), sf_L_a, get_pct(sf_L_a,'semi_final_pct'), sf_L_w)}</div>
  <div class="gap"></div>
  <div class="center-col">{final_col(f0, fp0, f1, fp1, winner, wp)}</div>
  <div class="gap"></div>
  <div class="col">{sf_col("Semi-Finals", sf_R_h, get_pct(sf_R_h,'semi_final_pct'), sf_R_a, get_pct(sf_R_a,'semi_final_pct'), sf_R_w)}</div>
  <div class="gap"></div>
  <div class="col">{stage_col("Quarter-Finals", qf_data(right_qf_order), qf_centers)}</div>
  <div class="gap"></div>
  <div class="col">{stage_col("Round of 16",  r16_data(right_r16_order), r16_centers)}</div>
  <div class="gap"></div>
  <div class="col">{stage_col("Round of 32",  r32_data(right_r32_order), r32_centers)}</div>
</div>"""

    components.html(html, height=700, scrolling=True)

# ════════════════════════════════
# VIEW 3 — TEAM PROFILE
# ════════════════════════════════
elif view == "🔍 Team Profile":
    st.markdown('<div class="section-title">Team Profile</div>', unsafe_allow_html=True)
    team_options = [f"{flag(t)} {t}" for t in df["team"].tolist()]
    selected_label = st.selectbox("Select a team", team_options)
    selected_team = df["team"][df["team"].apply(lambda t: f"{flag(t)} {t}") == selected_label].values[0]
    team_row = df[df["team"] == selected_team].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Win Tournament",      f"{team_row['win_pct']}%")
    with c2: st.metric("Reach Final",         f"{team_row['final_pct']}%")
    with c3: st.metric("Reach Semi-Final",    f"{team_row['semi_final_pct']}%")
    with c4: st.metric("Reach Quarter-Final", f"{team_row['quarter_final_pct']}%")
    cats = ["Win","Final","Semi-Final","Quarter-Final"]
    vals = [team_row["win_pct"],team_row["final_pct"],team_row["semi_final_pct"],team_row["quarter_final_pct"]]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                                     fillcolor="rgba(201,168,76,0.15)", line=dict(color=PRIMARY,width=2)))
    fig_r.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(visible=True,gridcolor=BORDER,color=TEXT_MUTED),
                                    angularaxis=dict(gridcolor=BORDER,color=TEXT)),
                         paper_bgcolor=BG, font_color=TEXT, showlegend=False, height=400,
                         margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig_r, use_container_width=True)
    fig_s = go.Figure(go.Bar(x=cats, y=vals, marker_color=[PRIMARY,WARNING,WARNING,WARNING],
                              text=[f"{v:.1f}%" for v in vals], textposition="outside"))
    fig_s.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                         yaxis_title="Probability (%)", height=350, margin=dict(l=20,r=20,t=20,b=20))
    fig_s.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(fig_s, use_container_width=True)

# ════════════════════════════════
# VIEW 4 — MONTE CARLO INSIGHTS
# ════════════════════════════════
elif view == "🎲 Monte Carlo Insights":
    st.markdown('<div class="section-title">Monte Carlo Insights</div>', unsafe_allow_html=True)
    df_plot = df.sort_values("win_pct", ascending=False).copy()
    df_plot["label"] = df_plot["team"].apply(lambda t: f"{flag(t)} {t}")
    fig_d = px.bar(df_plot, x="label", y="win_pct", color="win_pct",
                   color_continuous_scale=[[0,SURFACE2],[0.5,WARNING],[1,PRIMARY]])
    fig_d.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                         xaxis_title="", yaxis_title="Win Probability (%)",
                         coloraxis_showscale=False, height=400, margin=dict(l=20,r=20,t=20,b=100))
    fig_d.update_xaxes(tickangle=45, gridcolor=BORDER); fig_d.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(fig_d, use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**🌟 Dark Horses**")
        for t in insights["dark_horses"]:
            prob = round(t["semi_final_prob"]*100, 1)
            st.markdown(f"""<div class="insight-card">
                <strong style="color:{PRIMARY}">{flag(t['team'])} {t['team']}</strong><br>
                <span style="color:{TEXT_MUTED}">{prob}% chance of reaching Semi-Finals</span>
            </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("**⚡ Biggest Upsets**")
        for t in insights["biggest_upsets"]:
            prob = round(t["quarter_final_prob"]*100, 1)
            st.markdown(f"""<div class="insight-card">
                <strong style="color:{SECONDARY}">{flag(t['team'])} {t['team']}</strong><br>
                <span style="color:{TEXT_MUTED}">{prob}% chance of reaching Quarter-Finals</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════
# VIEW 5 — FEATURE IMPORTANCE
# ════════════════════════════════
elif view == "📊 Feature Importance":
    st.markdown('<div class="section-title">XGBoost Feature Importance</div>', unsafe_allow_html=True)
    st.caption("Features ranked by F-score — how often each feature was used to split nodes in the XGBoost trees.")
    importances = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=True)
    fig_i = px.bar(x=importances.values, y=importances.index, orientation="h",
                   color=importances.values, color_continuous_scale=[[0,SURFACE2],[0.5,WARNING],[1,PRIMARY]])
    fig_i.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                         xaxis_title="F-score", yaxis_title="",
                         coloraxis_showscale=False, height=500, margin=dict(l=20,r=40,t=20,b=20))
    fig_i.update_xaxes(gridcolor=BORDER); fig_i.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(fig_i, use_container_width=True)
    st.markdown(f"""<div class="insight-card">
        <strong style="color:{PRIMARY}">Model Performance</strong><br>
        <span style="color:{TEXT_MUTED}">Binary XGBoost classifier · Target: Home Win vs No Home Win ·
        Accuracy: 71% · Training set: 418 historical WC matches between 2026 qualified nations</span>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════
# VIEW 6 — MODEL VALIDATION
# ════════════════════════════════
elif view == "🧪 Model Validation":
    st.markdown('<div class="section-title">Model Validation — Predicted vs Actual</div>', unsafe_allow_html=True)

    if df_results.empty or df_results['actual_winner'].isna().all() or (df_results['actual_winner'] == '').all():
        st.info("No results yet — check back after the first match day.")
    else:
        # Filter to completed matches only
        completed = df_results[
            df_results['home_score'].notna() &
            (df_results['home_score'] != '')
        ].copy()

        # Compute correct predictions for knockout matches (group stage has no predicted_winner)
        knockout = completed[completed['stage'] != 'group_stage'].copy()
        if not knockout.empty:
            knockout['correct'] = knockout['actual_winner'] == knockout['predicted_winner']
            n_correct = knockout['correct'].sum()
            n_total   = len(knockout)
            accuracy  = round(n_correct / n_total * 100, 1) if n_total > 0 else 0

            # Accuracy metric
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Matches Predicted", n_total)
            with c2:
                st.metric("Correct Predictions", int(n_correct))
            with c3:
                st.metric("Accuracy", f"{accuracy}%")

            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

            # Results table
            display = knockout[['match_id','date','stage','home_team','away_team',
                                'home_score','away_score','actual_winner','predicted_winner','correct']].copy()
            display['correct'] = display['correct'].map({True: '✅', False: '❌'})
            display['stage'] = display['stage'].str.replace('_', ' ').str.title()
            display.columns = ['ID','Date','Stage','Home','Away','Home Score','Away Score',
                               'Actual Winner','Predicted Winner','Correct']
            st.dataframe(display, use_container_width=True, hide_index=True)

            # Upset tracker
            upsets = knockout[~knockout['correct']].copy()
            if not upsets.empty:
                st.markdown('<div class="section-title">⚡ Upsets — Model Got Wrong</div>', unsafe_allow_html=True)
                for _, row in upsets.iterrows():
                    predicted_prob = df[df['team'] == row['predicted_winner']]['win_prob'].values
                    actual_prob    = df[df['team'] == row['actual_winner']]['win_prob'].values
                    pred_p  = round(float(predicted_prob[0])*100, 1) if len(predicted_prob) else 0
                    actual_p = round(float(actual_prob[0])*100, 1) if len(actual_prob) else 0
                    st.markdown(f"""<div class="insight-card">
                        <strong style="color:{SECONDARY}">{flag(row['actual_winner'])} {row['actual_winner']}</strong>
                        beat <strong>{flag(row['predicted_winner'])} {row['predicted_winner']}</strong>
                        <br><span style="color:{TEXT_MUTED}">
                        Predicted: {row['predicted_winner']} ({pred_p}% win prob) ·
                        Actual: {row['actual_winner']} ({actual_p}% win prob)
                        </span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("Model has predicted all knockout matches correctly so far!")

        # Group stage results table
        group_completed = completed[completed['stage'] == 'group_stage']
        if not group_completed.empty:
            st.markdown('<div class="section-title">Group Stage Results</div>', unsafe_allow_html=True)
            display_g = group_completed[['match_id','date','group','home_team','away_team',
                                         'home_score','away_score']].copy()
            display_g.columns = ['ID','Date','Group','Home','Away','Home Score','Away Score']
            st.dataframe(display_g, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center;color:{TEXT_MUTED};font-size:0.8rem;">'
            "Built with XGBoost + Monte Carlo simulation · 10,000 tournament simulations · "
            "Training data: 418 historical WC matches between 2026 qualified nations</div>",
            unsafe_allow_html=True)
# NOTE: This append block is just for reference — see full file rebuild below !


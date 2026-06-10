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
    "Argentina":"🇦🇷","Algeria":"🇩🇿","Australia":"🇦🇺","Austria":"🇦🇹",
    "Belgium":"🇧🇪","Bosnia and Herzegovina":"🇧🇦","Brazil":"🇧🇷",
    "Cape Verde":"🇨🇻","Canada":"🇨🇦","Colombia":"🇨🇴","Croatia":"🇭🇷",
    "Czechia":"🇨🇿","Ivory Coast":"🇨🇮","Curaçao":"🇨🇼","DR Congo":"🇨🇩",
    "Ecuador":"🇪🇨","Egypt":"🇪🇬","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷",
    "Germany":"🇩🇪","Ghana":"🇬🇭","Haiti":"🇭🇹","Iran":"🇮🇷","Iraq":"🇮🇶",
    "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
    "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴","Panama":"🇵🇦",
    "Paraguay":"🇵🇾","Portugal":"🇵🇹","Qatar":"🇶🇦","Saudi Arabia":"🇸🇦",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Senegal":"🇸🇳","South Africa":"🇿🇦",
    "South Korea":"🇰🇷","Spain":"🇪🇸","Sweden":"🇸🇪","Switzerland":"🇨🇭",
    "Tunisia":"🇹🇳","Turkey":"🇹🇷","Uruguay":"🇺🇾","United States":"🇺🇸",
    "Uzbekistan":"🇺🇿",
}
def flag(team): return FLAGS.get(team, "🏳️")

st.markdown(f"""
<style>
    .stApp {{ background-color:{BG}; color:{TEXT}; }}
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

# ── Build group positions lookup ─────────────────────────────────────────────
gp = {}
for _, row in bp.iterrows():
    g, p, t = row['group'], int(row['position']), row['team']
    if g not in gp: gp[g] = {}
    gp[g][p] = t

# Best 8 third-place teams
thirds = bp[bp['position']==3].sort_values('probability', ascending=False)
thirds_list = [(row['team'], row['group']) for _, row in thirds.iterrows()]

third_slot_groups = {
    'ABCDF': set('ABCDF'), 'CDFGH': set('CDFGH'),
    'CEFHI': set('CEFHI'), 'EHIJK': set('EHIJK'),
    'AEHIJ': set('AEHIJ'), 'BEFIJ': set('BEFIJ'),
    'EFGIJ': set('EFGIJ'), 'DEIJL': set('DEIJL'),
}
compat = {slot: [] for slot in third_slot_groups}
for team, group in thirds_list:
    for slot, groups in third_slot_groups.items():
        if group in groups:
            compat[slot].append((team, group))

assigned_thirds = {}
used_teams = set()
for slot in sorted(compat, key=lambda s: len(compat[s])):
    for team, group in compat[slot]:
        if team not in used_teams:
            assigned_thirds[slot] = team
            used_teams.add(team)
            break
    if slot not in assigned_thirds:
        assigned_thirds[slot] = "TBD"

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

def get_pct(team, col):
    v = df[df['team']==team][col].values
    return round(float(v[0])*100, 1) if len(v) else 0.0

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(str(ASSETS / "banner.png"), use_container_width=True)
    except Exception:
        st.markdown(f'<div style="text-align:center;font-size:2rem;padding:1rem;">⚽</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    view = st.radio("Navigation",
        ["🏆 Hero","🗂 Tournament Bracket","🔍 Team Profile","🎲 Monte Carlo Insights","📊 Feature Importance"],
        label_visibility="collapsed")

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
# Official bracket: R32 → R16 → QF → SF → Final
# Left side = SF 101 branch, Right side = SF 102 branch
# ════════════════════════════════
elif view == "🗂 Tournament Bracket":
    st.markdown('<div class="section-title">Tournament Bracket — Predicted Path</div>', unsafe_allow_html=True)
    st.caption("Official 2026 bracket structure. Most likely team per slot based on 10,000 Monte Carlo simulations.")

    def tb(team, prob, highlight=False):
        f = flag(team)
        name = team[:14]+"…" if len(team)>15 else team
        cls = "tb hi" if highlight else "tb"
        return f'<div class="{cls}">{f} {name}<br><span class="prob">{prob:.1f}%</span></div>'

    def spacer(h): return f'<div style="height:{h}px"></div>'

    # LEFT side — SF 101: QF97(R16 89,90) and QF98(R16 93,94)
    # R32 order on left: 73,75 → R16 89, then 74,77 → R16 90, then 83,84 → R16 93, then 81,82 → R16 94
    left_r32_order = [73, 75, 74, 77, 83, 84, 81, 82]
    right_r32_order = [76, 78, 79, 80, 86, 88, 85, 87]

    left_r16_order  = [89, 90, 93, 94]
    right_r16_order = [91, 92, 95, 96]

    left_qf_order  = [97, 98]
    right_qf_order = [99, 100]

    def r32_html(order, side):
        items = []
        for i, mid in enumerate(order):
            h, a = r32_teams[mid]
            w = match_winners[mid]
            ph = get_pct(h, 'win_prob') * 100 if h != "TBD" else 0
            pa = get_pct(a, 'win_prob') * 100 if a != "TBD" else 0
            # Use quarter_final_pct as proxy for strength
            ph = get_pct(h, 'quarter_final_pct')
            pa = get_pct(a, 'quarter_final_pct')
            if i > 0 and i % 2 == 0:
                items.append(spacer(16))
            elif i > 0:
                items.append(spacer(6))
            items.append(tb(h, ph, h==w))
            items.append(spacer(4))
            items.append(tb(a, pa, a==w))
        return "".join(items)

    def r16_html(order):
        items = []
        for i, mid in enumerate(order):
            h, a = r16_matches[mid]
            w = match_winners[mid]
            ph = get_pct(h, 'semi_final_pct')
            pa = get_pct(a, 'semi_final_pct')
            if i > 0:
                items.append(spacer(28))
            items.append(tb(h, ph, h==w))
            items.append(spacer(4))
            items.append(tb(a, pa, a==w))
        return "".join(items)

    def qf_html(order):
        items = []
        for i, mid in enumerate(order):
            h, a = qf_matches[mid]
            w = match_winners[mid]
            ph = get_pct(h, 'final_pct')
            pa = get_pct(a, 'final_pct')
            if i > 0:
                items.append(spacer(80))
            items.append(tb(h, ph, h==w))
            items.append(spacer(4))
            items.append(tb(a, pa, a==w))
        return "".join(items)

    sf_L_h, sf_L_a = sf_matches[101]
    sf_R_h, sf_R_a = sf_matches[102]
    sf_L_w = match_winners[101]
    sf_R_w = match_winners[102]

    f0, f1 = final_teams
    fp0 = get_pct(f0, 'final_pct')
    fp1 = get_pct(f1, 'final_pct')
    wp  = get_pct(winner, 'win_pct')

    html = f"""
<style>
.bk {{ display:flex; align-items:flex-start; justify-content:center; background:#13131F;
       padding:20px 8px; gap:0; font-family:Arial,sans-serif; overflow-x:auto; min-height:560px; }}
.col {{ display:flex; flex-direction:column; min-width:130px; }}
.lbl {{ color:#C9A84C; font-size:9px; font-weight:bold; text-align:center;
        letter-spacing:0.5px; margin-bottom:10px; text-transform:uppercase; padding-top:2px; }}
.tb {{ background:rgba(45,35,10,0.95); border:1px solid #3a3010; border-radius:4px;
       padding:5px 7px; font-size:8px; color:#F0F0FA; text-align:center;
       max-width:126px; line-height:1.35; }}
.tb.hi {{ background:rgba(110,85,20,0.98); border-color:#C9A84C; }}
.prob {{ color:#9090B0; font-size:7px; }}
.gap {{ width:14px; }}
.center {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
           min-width:152px; padding:0 6px; padding-top:2px; }}
.clbl {{ color:#C9A84C; font-size:10px; font-weight:bold; text-align:center;
         letter-spacing:0.5px; margin-bottom:8px; text-transform:uppercase; }}
.fin {{ background:rgba(110,85,20,0.98); border:1px solid #C9A84C; border-radius:4px;
        padding:7px 10px; font-size:9px; color:#F0F0FA; text-align:center; width:144px; }}
.win {{ background:#C9A84C; border:2px solid #D4A017; border-radius:6px;
        padding:10px 12px; font-size:10px; font-weight:bold; color:#13131F;
        text-align:center; width:144px; margin:6px 0; }}
.win .wp {{ font-size:8px; color:#3a2a05; margin-top:3px; }}
</style>
<div class="bk">
  <div class="col"><div class="lbl">Round of 32</div>{r32_html(left_r32_order,'left')}</div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Round of 16</div><div style="height:14px"></div>{r16_html(left_r16_order)}</div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Quarter-Finals</div><div style="height:46px"></div>{qf_html(left_qf_order)}</div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Semi-Finals</div><div style="height:110px"></div>
    {tb(sf_L_h, get_pct(sf_L_h,'semi_final_pct'), sf_L_h==sf_L_w)}
    {spacer(4)}{tb(sf_L_a, get_pct(sf_L_a,'semi_final_pct'), sf_L_a==sf_L_w)}
  </div>
  <div class="gap"></div>
  <div class="center">
    <div class="clbl">⚽ Final</div>
    <div class="fin">{flag(f0)} {f0}<br><span style="color:#9090B0;font-size:7px;">{fp0}%</span></div>
    <div class="win">🏆 {flag(winner)} {winner}<div class="wp">{wp}% win probability</div></div>
    <div class="fin">{flag(f1)} {f1}<br><span style="color:#9090B0;font-size:7px;">{fp1}%</span></div>
  </div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Semi-Finals</div><div style="height:110px"></div>
    {tb(sf_R_h, get_pct(sf_R_h,'semi_final_pct'), sf_R_h==sf_R_w)}
    {spacer(4)}{tb(sf_R_a, get_pct(sf_R_a,'semi_final_pct'), sf_R_a==sf_R_w)}
  </div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Quarter-Finals</div><div style="height:46px"></div>{qf_html(right_qf_order)}</div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Round of 16</div><div style="height:14px"></div>{r16_html(right_r16_order)}</div>
  <div class="gap"></div>
  <div class="col"><div class="lbl">Round of 32</div>{r32_html(right_r32_order,'right')}</div>
</div>"""
    components.html(html, height=600, scrolling=True)

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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center;color:{TEXT_MUTED};font-size:0.8rem;">'
            "Built with XGBoost + Monte Carlo simulation · 10,000 tournament simulations · "
            "Training data: 418 historical WC matches between 2026 qualified nations</div>",
            unsafe_allow_html=True)

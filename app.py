import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from dotenv import load_dotenv

# ── Path setup — works both locally and on Streamlit Cloud ──────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
load_dotenv()

# ── Team colors ─────────────────────────────────────────────────
TEAM_COLORS = {
    'Mumbai Indians':                '#004BA0',
    'Chennai Super Kings':           '#FDB913',
    'Royal Challengers Bangalore':   '#EC1C24',
    'Kolkata Knight Riders':         '#3A225D',
    'Sunrisers Hyderabad':           '#F7A721',
    'Rajasthan Royals':              '#EA1A85',
    'Delhi Capitals':                '#00AAE7',
    'Delhi Daredevils':              '#00AAE7',
    'Punjab Kings':                  '#AAAAAA',
    'Kings XI Punjab':               '#AAAAAA',
    'Gujarat Titans':                '#1C4E80',
    'Lucknow Super Giants':          '#A0C4FF',
    'Kochi Tuskers Kerala':          '#F58220',
    'Pune Warriors':                 '#5B2D8E',
    'Rising Pune Supergiant':        '#6F2DA8',
    'Gujarat Lions':                 '#E8461A',
}

def get_color(team):
    return TEAM_COLORS.get(team, '#888888')

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics",
    page_icon="🏏",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border-left: 4px solid #004BA0;
}
.big-number {
    font-size: 2rem;
    font-weight: bold;
    color: #004BA0;
}
</style>
""", unsafe_allow_html=True)

# ── Load data and model ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/deliveries.csv')

@st.cache_resource
def load_model():
    return joblib.load('src/win_prob_model.pkl')

df = load_data()
model = load_model()

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🏏 IPL Analytics")
st.sidebar.caption("1,226 matches · 291,399 deliveries")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Team Performance",
    "Player Stats",
    "Win Probability",
    "What-If Simulator"
])

# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🏏 IPL Analytics Dashboard")
    st.caption("Ball-by-ball analysis of 1,226 IPL matches (2008–2023)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total matches", f"{df['match_id'].nunique():,}")
    col2.metric("Total deliveries", f"{len(df):,}")
    col3.metric("Unique teams", df['batting_team'].nunique())
    col4.metric("Unique venues", df['venue'].nunique())

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Key finding — toss advantage is a myth")
        toss = df.drop_duplicates('match_id')[['match_id','toss_winner','winner']].copy()
        toss['toss_helped'] = (toss['toss_winner'] == toss['winner']).astype(int)
        pct = toss['toss_helped'].mean() * 100
        st.metric("Toss winner won the match", f"{pct:.1f}%",
                  delta="Basically a coin flip", delta_color="off")
        st.caption("Close to 50% across all IPL seasons = no real toss advantage")

    with col_b:
        st.subheader("Toss decision trend")
        decisions = df.drop_duplicates('match_id')['toss_decision'].value_counts().reset_index()
        fig_toss = px.pie(decisions, names='toss_decision', values='count',
                          color_discrete_sequence=['#004BA0','#FDB913'])
        fig_toss.update_layout(margin=dict(t=20,b=20,l=20,r=20), height=250)
        st.plotly_chart(fig_toss, use_container_width=True)

    st.divider()
    st.subheader("Average runs per over")
    over_runs = df.groupby('over')['runs_total'].mean().reset_index()
    over_runs.columns = ['over', 'avg_runs']
    fig_over = px.bar(over_runs, x='over', y='avg_runs',
                      labels={'over': 'Over', 'avg_runs': 'Avg runs per ball'},
                      color='avg_runs', color_continuous_scale='Blues')
    st.plotly_chart(fig_over, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — TEAM PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif page == "Team Performance":
    st.title("Team Performance")

    # Total wins with team colors
    st.subheader("Total wins by team")
    wins = df.drop_duplicates('match_id')['winner'].value_counts().reset_index()
    wins.columns = ['team', 'wins']
    wins = wins[wins['team'] != 'No result']
    wins['color'] = wins['team'].apply(get_color)

    fig_wins = go.Figure(go.Bar(
        x=wins['team'],
        y=wins['wins'],
        marker_color=wins['color'],
        text=wins['wins'],
        textposition='outside'
    ))
    fig_wins.update_layout(
        xaxis_tickangle=-45,
        height=500,
        plot_bgcolor='white',
        yaxis_title='Total wins'
    )
    st.plotly_chart(fig_wins, use_container_width=True)

    st.divider()

    # Team selector for detailed stats
    st.subheader("Team deep dive")
    all_teams = sorted([t for t in df['batting_team'].unique() if t in TEAM_COLORS])
    selected_team = st.selectbox("Select a team", all_teams)

    team_df = df[df['batting_team'] == selected_team]
    team_color = get_color(selected_team)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total matches batted", team_df['match_id'].nunique())
    col2.metric("Total runs scored", f"{team_df['runs_batter'].sum():,}")
    col3.metric("Total wickets lost", team_df['is_wicket'].sum())

    # Run rate by over for selected team
    team_over = team_df.groupby('over')['runs_total'].mean().reset_index()
    fig_team = px.bar(team_over, x='over', y='runs_total',
                      title=f'{selected_team} — avg runs per over',
                      labels={'runs_total': 'Avg runs', 'over': 'Over'},
                      color_discrete_sequence=[team_color])
    st.plotly_chart(fig_team, use_container_width=True)

    st.divider()

    # Venue stats
    st.subheader("Top venues by matches played")
    venue_counts = df.drop_duplicates('match_id')['venue'].value_counts().head(10).reset_index()
    venue_counts.columns = ['venue', 'matches']
    fig_venue = px.bar(venue_counts, x='matches', y='venue', orientation='h',
                       color='matches', color_continuous_scale='Oranges',
                       labels={'matches': 'Matches', 'venue': 'Venue'})
    st.plotly_chart(fig_venue, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — PLAYER STATS
# ════════════════════════════════════════════════════════════════
elif page == "Player Stats":
    st.title("Player Statistics")

    tab1, tab2 = st.tabs(["🏏 Batting", "🎯 Bowling"])

    with tab1:
        st.subheader("Top run scorers")
        n = st.slider("Show top N batters", 5, 20, 10)
        batters = df.groupby('batter').agg(
            runs=('runs_batter', 'sum'),
            balls=('runs_batter', 'count'),
            matches=('match_id', 'nunique')
        ).reset_index()
        batters['strike_rate'] = (batters['runs'] / batters['balls'] * 100).round(1)
        batters['average'] = (batters['runs'] / batters['matches']).round(1)
        batters = batters.sort_values('runs', ascending=False).head(n)

        fig_bat = px.bar(batters, x='batter', y='runs',
                         hover_data=['strike_rate', 'average', 'matches'],
                         color='strike_rate',
                         color_continuous_scale='RdYlGn',
                         labels={'runs': 'Total runs', 'batter': 'Batter',
                                 'strike_rate': 'Strike rate'})
        fig_bat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bat, use_container_width=True)
        st.caption("Color = strike rate (green = higher strike rate)")
        st.dataframe(batters.reset_index(drop=True), use_container_width=True)

    with tab2:
        st.subheader("Top wicket takers")
        n2 = st.slider("Show top N bowlers", 5, 20, 10)
        bowlers = df.groupby('bowler').agg(
            wickets=('is_wicket', 'sum'),
            balls=('is_wicket', 'count'),
            runs_given=('runs_total', 'sum')
        ).reset_index()
        bowlers['economy'] = (bowlers['runs_given'] / bowlers['balls'] * 6).round(2)
        bowlers = bowlers[bowlers['balls'] >= 120]
        bowlers = bowlers.sort_values('wickets', ascending=False).head(n2)

        fig_bowl = px.bar(bowlers, x='bowler', y='wickets',
                          hover_data=['economy'],
                          color='economy',
                          color_continuous_scale='RdYlGn_r',
                          labels={'wickets': 'Wickets', 'bowler': 'Bowler',
                                  'economy': 'Economy'})
        fig_bowl.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bowl, use_container_width=True)
        st.caption("Color = economy rate (green = lower economy = better)")
        st.dataframe(bowlers.reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — WIN PROBABILITY
# ════════════════════════════════════════════════════════════════
elif page == "Win Probability":
    st.title("Win Probability — Historical Match")
    st.caption("Select a match to see win probability change ball by ball")

    matches = sorted(df['match_id'].unique())
    selected_match = st.selectbox("Select match ID", matches[:200])

    match_df = df[(df['match_id'] == selected_match) & (df['inning'] == 2)].copy()
    match_df = match_df.sort_values(['over','ball']).reset_index(drop=True)

    if len(match_df) == 0:
        st.warning("No 2nd innings data for this match")
    else:
        inn1_runs = df[(df['match_id'] == selected_match) & (df['inning'] == 1)]['runs_total'].sum()
        target = inn1_runs + 1
        batting_team = match_df['batting_team'].iloc[0]
        winner = match_df['winner'].iloc[0]
        team_color = get_color(batting_team)

        col1, col2, col3 = st.columns(3)
        col1.metric("Target", target)
        col2.metric("Chasing team", batting_team)
        col3.metric("Result", winner)

        match_df['runs_so_far'] = match_df['runs_total'].cumsum()
        match_df['wickets_so_far'] = match_df['is_wicket'].cumsum()
        match_df['balls_bowled'] = range(1, len(match_df) + 1)
        match_df['balls_remaining'] = 120 - match_df['balls_bowled']
        match_df['runs_needed'] = target - match_df['runs_so_far']
        match_df['wickets_remaining'] = 10 - match_df['wickets_so_far']
        match_df['required_run_rate'] = (match_df['runs_needed'] / match_df['balls_remaining'].clip(1)) * 6
        match_df['current_run_rate'] = (match_df['runs_so_far'] / match_df['balls_bowled']) * 6
        match_df['run_rate_diff'] = match_df['current_run_rate'] - match_df['required_run_rate']
        match_df['phase'] = pd.cut(match_df['over'], bins=[-1,5,14,19], labels=[0,1,2]).astype(int)

        features = ['runs_needed','balls_remaining','wickets_remaining',
                    'required_run_rate','current_run_rate','run_rate_diff','phase']

        probs = model.predict_proba(match_df[features])[:, 1]
        match_df['win_prob'] = probs
        match_df['ball_num'] = range(1, len(match_df) + 1)

        # Mark wicket balls
        wicket_balls = match_df[match_df['is_wicket'] == 1]

        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=match_df['ball_num'],
            y=match_df['win_prob'],
            mode='lines',
            name='Win probability',
            line=dict(color=team_color, width=2)
        ))
        fig_prob.add_trace(go.Scatter(
            x=wicket_balls['ball_num'],
            y=wicket_balls['win_prob'],
            mode='markers',
            name='Wicket',
            marker=dict(color='red', size=10, symbol='x')
        ))
        fig_prob.add_hline(y=0.5, line_dash='dash', line_color='gray',
                           annotation_text="50%")
        fig_prob.update_layout(
            title=f'Win probability — {batting_team} chasing {target}',
            xaxis_title='Ball number',
            yaxis_title='Win probability',
            yaxis_range=[0, 1],
            height=450
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.caption("Red X markers = wickets. Line color = chasing team color.")

# ════════════════════════════════════════════════════════════════
# PAGE 5 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════
elif page == "What-If Simulator":
    st.title("⚡ What-If Match Simulator")
    st.caption("Change the match situation and see predicted win probability live")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Match situation")
        target = st.number_input("Target (runs)", min_value=50, max_value=300, value=175)
        runs_scored = st.slider("Runs scored so far", 0, int(target)-1, 80)
        balls_bowled = st.slider("Balls bowled (of 120)", 1, 119, 60)
        wickets_fallen = st.slider("Wickets fallen", 0, 9, 3)

    with col2:
        st.subheader("Chasing team")
        team_options = sorted(TEAM_COLORS.keys())
        selected_team = st.selectbox("Select chasing team", team_options)
        team_color = get_color(selected_team)
        st.color_picker("Team color", value=team_color, disabled=True)

    runs_needed = target - runs_scored
    balls_remaining = 120 - balls_bowled
    wickets_remaining = 10 - wickets_fallen
    rrr = (runs_needed / max(balls_remaining, 1)) * 6
    crr = (runs_scored / max(balls_bowled, 1)) * 6
    rr_diff = crr - rrr
    over = balls_bowled // 6
    phase = 0 if over <= 5 else (1 if over <= 14 else 2)

    features_input = pd.DataFrame([[
        runs_needed, balls_remaining, wickets_remaining,
        rrr, crr, rr_diff, phase
    ]], columns=['runs_needed','balls_remaining','wickets_remaining',
                 'required_run_rate','current_run_rate','run_rate_diff','phase'])

    win_prob = model.predict_proba(features_input)[0][1]

    st.divider()
    col3, col4, col5 = st.columns(3)
    col3.metric("Win probability", f"{win_prob*100:.1f}%")
    col4.metric("Required run rate", f"{rrr:.2f}")
    col5.metric("Current run rate", f"{crr:.2f}")

    # Color coded verdict
    if win_prob >= 0.7:
        st.success(f"✅ {selected_team} in strong position — {win_prob*100:.1f}% win probability")
    elif win_prob >= 0.4:
        st.warning(f"⚠️ Match evenly poised — {win_prob*100:.1f}% win probability")
    else:
        st.error(f"❌ {selected_team} under pressure — {win_prob*100:.1f}% win probability")

    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_prob * 100,
        title={'text': f"{selected_team} Win Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': team_color},
            'steps': [
                {'range': [0, 40], 'color': '#FFCCCC'},
                {'range': [40, 60], 'color': '#FFF3CC'},
                {'range': [60, 100], 'color': '#CCFFCC'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    st.subheader("Match situation summary")
    summary_df = pd.DataFrame({
        'Metric': ['Runs needed', 'Balls remaining', 'Wickets remaining',
                   'Required run rate', 'Current run rate', 'Run rate difference'],
        'Value': [runs_needed, balls_remaining, wickets_remaining,
                  f"{rrr:.2f}", f"{crr:.2f}", f"{rr_diff:.2f}"]
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

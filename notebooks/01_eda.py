# %% Setup
import os
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

os.chdir(r'C:\Users\Lenovo\ipl-analytics')
load_dotenv()

df = pd.read_csv('data/processed/deliveries.csv')
print(f"Loaded {len(df):,} deliveries")

# Create folder to save charts
os.makedirs('assets/charts', exist_ok=True)

# %% Question 1 — Does winning the toss help?
toss = df.drop_duplicates('match_id')[['match_id','toss_winner','winner']].copy()
toss['toss_helped'] = (toss['toss_winner'] == toss['winner']).astype(int)
pct = toss['toss_helped'].mean() * 100
print(f"\nToss winner won the match: {pct:.1f}% of the time")
print("(Close to 50% = toss advantage is a myth)")

# %% Question 2 — Average runs per over (combined)
over_runs = df.groupby('over')['runs_total'].mean().reset_index()
over_runs.columns = ['over', 'avg_runs']
fig1 = px.bar(over_runs, x='over', y='avg_runs',
              title='Average runs per over across all IPL matches',
              labels={'over': 'Over number', 'avg_runs': 'Avg runs'},
              color='avg_runs', color_continuous_scale='Blues')
fig1.write_image('assets/charts/01_avg_runs_per_over.png', width=1000, height=500)
print("Chart 1 saved")

# %% Question 3 — Top 10 run scorers
batters = df.groupby('batter')['runs_batter'].sum().reset_index()
batters = batters.sort_values('runs_batter', ascending=False).head(10)
fig2 = px.bar(batters, x='batter', y='runs_batter',
              title='Top 10 IPL run scorers (all time)',
              labels={'runs_batter': 'Total runs', 'batter': 'Batter'},
              color='runs_batter', color_continuous_scale='Oranges')
fig2.write_image('assets/charts/02_top_run_scorers.png', width=1000, height=500)
print("Chart 2 saved")

# %% Question 4 — Top 10 wicket takers
wickets = df[df['is_wicket'] == 1].groupby('bowler')['is_wicket'].sum().reset_index()
wickets = wickets.sort_values('is_wicket', ascending=False).head(10)
fig3 = px.bar(wickets, x='bowler', y='is_wicket',
              title='Top 10 IPL wicket takers (all time)',
              labels={'is_wicket': 'Wickets', 'bowler': 'Bowler'},
              color='is_wicket', color_continuous_scale='Reds')
fig3.write_image('assets/charts/03_top_wicket_takers.png', width=1000, height=500)
print("Chart 3 saved")

# %% Question 5 — Toss decision trends
decisions = df.drop_duplicates('match_id')['toss_decision'].value_counts().reset_index()
fig4 = px.pie(decisions, names='toss_decision', values='count',
              title='Toss decision — bat vs field',
              color_discrete_sequence=px.colors.qualitative.Set2)
fig4.write_image('assets/charts/04_toss_decisions.png', width=800, height=500)
print("Chart 4 saved")

# %% Question 6 — Most successful teams
wins = df.drop_duplicates('match_id')['winner'].value_counts().reset_index()
wins.columns = ['team', 'wins']
wins = wins[wins['team'] != 'No result'].head(10)
fig5 = px.bar(wins, x='team', y='wins',
              title='Most wins by team (all time)',
              labels={'wins': 'Total wins', 'team': 'Team'},
              color='wins', color_continuous_scale='Teal')
fig5.write_image('assets/charts/05_team_wins.png', width=1000, height=500)
print("Chart 5 saved")

# %% Question 7 — Runs by over split by inning
over_inning = df.groupby(['over','inning'])['runs_total'].mean().reset_index()
over_inning['inning'] = over_inning['inning'].astype(str)
fig6 = px.line(over_inning, x='over', y='runs_total', color='inning',
               title='Average runs per over — inning 1 vs inning 2',
               labels={'runs_total': 'Avg runs', 'over': 'Over', 'inning': 'Inning'},
               color_discrete_sequence=['#1f77b4','#ff7f0e'])
fig6.write_image('assets/charts/06_runs_by_inning.png', width=1000, height=500)
print("Chart 6 saved")

# %% Question 8 — Wickets per over
wickets_over = df.groupby('over')['is_wicket'].sum().reset_index()
fig7 = px.bar(wickets_over, x='over', y='is_wicket',
              title='Total wickets by over — when do bowlers strike?',
              labels={'is_wicket': 'Total wickets', 'over': 'Over'},
              color='is_wicket', color_continuous_scale='Purples')
fig7.write_image('assets/charts/07_wickets_per_over.png', width=1000, height=500)
print("Chart 7 saved")

# %% Question 9 — Top venues by matches
venue_counts = df.drop_duplicates('match_id')['venue'].value_counts().head(10).reset_index()
venue_counts.columns = ['venue', 'matches']
fig8 = px.bar(venue_counts, x='matches', y='venue', orientation='h',
              title='Top 10 IPL venues by number of matches',
              labels={'matches': 'Matches played', 'venue': 'Venue'},
              color='matches', color_continuous_scale='Greens')
fig8.write_image('assets/charts/08_top_venues.png', width=1000, height=500)
print("Chart 8 saved")

print("\nAll 8 charts saved to assets/charts/")


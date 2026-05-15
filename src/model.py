# %% Setup
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import joblib

os.chdir(r'C:\Users\Lenovo\ipl-analytics')
load_dotenv()

df = pd.read_csv('data/processed/deliveries.csv')
print(f"Loaded {len(df):,} deliveries")

# %% Build match state features
# We only care about 2nd innings — predicting if chasing team wins
df2 = df[df['inning'] == 2].copy()
df2 = df2.sort_values(['match_id', 'over', 'ball']).reset_index(drop=True)

# Cumulative runs and wickets at each ball
df2['runs_so_far'] = df2.groupby('match_id')['runs_total'].cumsum()
df2['wickets_so_far'] = df2.groupby('match_id')['is_wicket'].cumsum()
df2['balls_bowled'] = df2.groupby('match_id').cumcount() + 1
df2['balls_remaining'] = 120 - df2['balls_bowled']

# Get innings 1 total for each match
inn1 = df[df['inning'] == 1].groupby('match_id')['runs_total'].sum().reset_index()
inn1.columns = ['match_id', 'target']
inn1['target'] = inn1['target'] + 1  # need 1 more than inn1 score

df2 = df2.merge(inn1, on='match_id', how='left')
df2 = df2.dropna(subset=['target'])

# Calculate match state features
df2['runs_needed'] = df2['target'] - df2['runs_so_far']
df2['wickets_remaining'] = 10 - df2['wickets_so_far']
df2['required_run_rate'] = (df2['runs_needed'] / df2['balls_remaining'].clip(1)) * 6
df2['current_run_rate'] = (df2['runs_so_far'] / df2['balls_bowled']) * 6
df2['run_rate_diff'] = df2['current_run_rate'] - df2['required_run_rate']

# Match phase
df2['phase'] = pd.cut(df2['over'],
    bins=[-1, 5, 14, 19],
    labels=[0, 1, 2]).astype(int)

# Target — did batting team win?
df2['target_won'] = (df2['winner'] == df2['batting_team']).astype(int)

print(f"\nInning 2 deliveries: {len(df2):,}")
print(f"Win rate in inning 2: {df2['target_won'].mean()*100:.1f}%")

# %% Define features and target
features = [
    'runs_needed',
    'balls_remaining',
    'wickets_remaining',
    'required_run_rate',
    'current_run_rate',
    'run_rate_diff',
    'phase'
]

X = df2[features]
y = df2['target_won']
groups = df2['match_id']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Class balance: {y.mean()*100:.1f}% wins")

# %% Train model with GroupKFold
# CRITICAL: GroupKFold prevents data leakage
# Balls from same match must not appear in both train and test
print("\nTraining model with GroupKFold cross validation...")
gkf = GroupKFold(n_splits=5)
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

auc_scores = []
brier_scores = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    auc_scores.append(auc)
    brier_scores.append(brier)
    print(f"  Fold {fold+1} — ROC-AUC: {auc:.3f}  Brier: {brier:.3f}")

print(f"\nMean ROC-AUC: {np.mean(auc_scores):.3f} (+/- {np.std(auc_scores):.3f})")
print(f"Mean Brier score: {np.mean(brier_scores):.3f}")
print("(Brier score: lower is better, 0 = perfect, 0.25 = random)")

# %% Train final model on all data
print("\nTraining final model on full dataset...")
model.fit(X, y)

# %% Feature importance
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

fig = px.bar(importance_df, x='importance', y='feature',
             orientation='h',
             title='Feature importance — win probability model',
             labels={'importance': 'Importance', 'feature': 'Feature'})
os.makedirs('assets/charts', exist_ok=True)
fig.write_image('assets/charts/09_feature_importance.png', width=900, height=500)
print("\nFeature importance chart saved")

# %% Save model
os.makedirs('src', exist_ok=True)
joblib.dump(model, 'src/win_prob_model.pkl')
print("Model saved to src/win_prob_model.pkl")

# %% Quick sanity check — predict a match situation
print("\n--- Sanity check ---")
situations = [
    # [runs_needed, balls_remaining, wickets_remaining, rrr, crr, rr_diff, phase]
    [20,  24, 8, 5.0,  8.0,  3.0,  2],  # Easy chase — should be high win prob
    [60,  24, 8, 15.0, 8.0, -7.0,  2],  # Very hard chase — should be low
    [30,  30, 5, 6.0,  7.0,  1.0,  2],  # Balanced — should be around 50%
]
labels = ["Easy chase (20 off 24, 8 wkts)", 
          "Hard chase (60 off 24, 8 wkts)",
          "Balanced (30 off 30, 5 wkts)"]

for situation, label in zip(situations, labels):
    prob = model.predict_proba([situation])[0][1]
    print(f"{label}: {prob*100:.1f}% win probability")

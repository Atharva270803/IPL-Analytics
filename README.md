# 🏏 IPL Analytics — Win Probability & Match Intelligence

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![ML](https://img.shields.io/badge/Model-GradientBoosting-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

## Live demo
🔗 [Coming soon — deploy link here]

---

## What this project does
IPL commentators estimate win probability subjectively. This project 
builds a data-driven win probability model using ball-by-ball data 
from 1,226 IPL matches (2008–2023), and delivers insights through 
an interactive Streamlit dashboard.

---

## Screenshots

### Overview dashboard
![Overview](assets/charts/app_overview.png)

### Team performance with team colors
![Teams](assets/charts/app_teams.png)

### What-If match simulator
![Simulator](assets/charts/app_simulator.png)

---

## Key findings
- **Toss advantage is a myth** — toss winner won only 50.6% of matches 
  (statistically indistinguishable from a coin flip)
- **Death overs dominate** — overs 17-19 account for the most wickets 
  and highest run rates
- **Required run rate is the strongest predictor** of win probability 
  (49% feature importance), confirming that match momentum is best 
  captured through pressure metrics

---

## What I built

| Component | Description |
|---|---|
| ETL Pipeline | Parsed 1,226 Cricsheet YAML files into PostgreSQL |
| EDA | 8 business questions answered with Plotly visualisations |
| ML Model | Gradient Boosting win probability model (ROC-AUC 0.864) |
| Dashboard | 5-page Streamlit app with team colors and live simulator |

---

## Model performance

| Metric | Score | Meaning |
|---|---|---|
| ROC-AUC | 0.864 | Ranks outcomes correctly 86% of the time |
| Brier score | 0.151 | Good probability calibration |
| CV folds | 5 (GroupKFold) | Prevents data leakage across matches |

### Why GroupKFold matters
Regular train-test split would put balls from the same match in both 
train and test sets — the model would effectively see the future. 
GroupKFold ensures entire matches are held out, which is the honest 
evaluation.

### Known limitation
The model assigns high importance to required run rate (49%) and may 
underweight wicket value in pressure situations. Future improvement: 
add match phase interaction features.

---

## EDA highlights

![Runs per over](assets/charts/01_avg_runs_per_over.png)
![Top run scorers](assets/charts/02_top_run_scorers.png)
![Top wicket takers](assets/charts/03_top_wicket_takers.png)
![Team wins](assets/charts/05_team_wins.png)

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| PostgreSQL 15 | Database for 291,399 deliveries |
| SQLAlchemy | Database ORM |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML model and evaluation |
| Plotly | Interactive visualisations |
| Streamlit | Web dashboard |
| SHAP | Model explainability (planned) |

---

## How to run locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ipl-analytics.git
cd ipl-analytics

# 2. Create conda environment
conda create -n ipl-analytics python=3.11
conda activate ipl-analytics
pip install -r requirements.txt

# 3. Set up PostgreSQL
# Create a database called ipl_db
# Add credentials to .env file:
# DATABASE_URL=postgresql://postgres:password@localhost:5432/ipl_db

# 4. Download data
# Get IPL YAML files from cricsheet.org/downloads
# Place in data/raw/

# 5. Run ETL
python src/etl.py

# 6. Run EDA and model
python notebooks/01_eda.py
python src/model.py

# 7. Launch app
streamlit run app.py
```

---

## Project structure
ipl-analytics/
├── data/
│   ├── raw/           ← Cricsheet YAML files (not committed)
│   └── processed/     ← Cleaned CSV + charts
├── notebooks/
│   └── 01_eda.py      ← EDA and visualisations
├── src/
│   ├── etl.py         ← YAML to PostgreSQL pipeline
│   ├── features.py    ← Feature engineering
│   ├── model.py       ← ML model training
│   └── db.py          ← Database connection
├── assets/
│   └── charts/        ← PNG charts for README
├── app.py             ← Streamlit dashboard
├── requirements.txt
└── README.md

---

## What I learned
- GroupKFold is essential for sports data — regular CV leaks match 
  context across train and test sets
- Brier score matters for probability models — calibration is as 
  important as discrimination (ROC-AUC)
- ETL from raw YAML across 1,226 files surfaces real data quality 
  issues that Kaggle CSVs hide
- Required run rate explains more variance in win probability than 
  wickets — pressure metrics dominate in T20

---

## Interview talking points
- Why GroupKFold instead of train_test_split?
- What does Brier score measure vs ROC-AUC?
- What did the toss advantage analysis find?
- How did you handle data leakage in time-series sports data?

---

*Built as part of a data science portfolio targeting analytics roles 
at BFSI and product companies.*
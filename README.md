# FIFA World Cup 2026 — ML Winner Prediction

A machine learning project that predicts the FIFA World Cup 2026 winner using match-by-match simulation powered by XGBoost and Monte Carlo methods.

---

## How It Works

Rather than picking a single predicted winner, the model simulates the entire tournament **10,000 times**. Each match outcome is sampled probabilistically from the model's predictions, producing a full **win probability distribution** across all 48 teams.

---

## Project Structure

```
wc2026-predictor/
│
├── data/
│   ├── raw/
│   │   ├── wc_matches_1930_2022.csv          # Historical WC match results (training data)
│   │   ├── elo_ratings_wc2026.csv            # Elo ratings for all 48 qualified teams
│   │   ├── team_features_train.csv           # ML-ready team features (historical WCs)
│   │   ├── team_features_test.csv            # ML-ready team features (2026 test set)
│   │   ├── wc2026_host_cities.csv            # Host city and venue info
│   │   ├── wc2026_matches.csv                # All 104 fixtures and bracket structure
│   │   ├── wc2026_teams.csv                  # All 48 qualified teams
│   │   └── wc2026_tournament_stages.csv      # Stage progression logic
│   └── processed/
│       ├── df_elo_clean.csv                  # Elo ratings filtered to 2026-05-27 snapshot
│       ├── df_host_cities_clean.csv          # Host cities with region clusters
│       ├── df_matches_clean.csv              # Cleaned historical matches
│       ├── df_matches_features.csv           # Historical matches with Elo features (training ready)
│       ├── df_matches_2026_clean.csv         # Raw 2026 fixtures
│       ├── df_matches_2026_features.csv      # 2026 fixtures with all features (simulation ready)
│       ├── df_stages_2026_clean.csv          # Tournament stage progression
│       ├── df_teams_2026_clean.csv           # All 48 qualified teams (placeholders resolved)
│       ├── df_test_clean.csv                 # 2026 team features test set
│       └── df_train_clean.csv                # Historical team features training set
│
├── notebooks/
│   ├── 01_EDA.ipynb                      # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb      # Build match-level feature matrix
│   ├── 03_model_training.ipynb           # Train and evaluate XGBoost classifier
│   └── 04_simulation.ipynb               # Monte Carlo tournament simulation
│
├── src/
│   ├── features.py                       # Feature engineering functions
│   ├── model.py                          # XGBoost training and prediction logic
│   └── simulation.py                     # Monte Carlo simulation engine
│
└── README.md
```

---

## Datasets

| # | Dataset | Source |
|---|---------|--------|
| 1 | FIFA WC 1930–2022 All Match Dataset | [Kaggle](https://www.kaggle.com/datasets/jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset) |
| 2 | 2026 WC Historical Elo Ratings | [Kaggle](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings) |
| 3 | FIFA World Cup Team Dataset | [Kaggle](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset) |
| 4 | 2026 WC Match Data | [Kaggle](https://www.kaggle.com/datasets/areezvisram12/fifa-world-cup-2026-match-data-unofficial) |

---

## Features per Match

- Elo rating of each team (and the difference)
- FIFA ranking of each team
- Historical head-to-head record
- Goals scored / conceded in recent matches (form)
- World Cup experience (number of past tournaments)
- Confederation (UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC)
- Host continent advantage
- Tournament stage (group stage vs knockout)

---

## Model

**XGBoost** classifier trained on historical World Cup match data (1930–2022).

- **Group stage target:** Win / Draw / Loss (3-class classification)
- **Knockout stage target:** Win / Loss (draws resolved via Elo-weighted penalty probability)

---

## Monte Carlo Simulation

```
1. Group Stage      → Simulate all 48 group matches → rank teams → advance top 2 per group + best 8 third-place teams
2. Round of 32      → Simulate 16 matches → advance 32 winners  
3. Round of 16      → Simulate 16 matches → advance 16 winners
4. Quarter-Finals   → Simulate 8 matches  → advance 8 winners
5. Semi-Finals      → Simulate 4 matches  → advance 4 winners
6. Final            → Simulate 1 match    → tournament winner
```

Repeat **10,000 times**. Final win probability per team = simulations won ÷ 10,000.

---

## Stack

```
pandas        Data loading and wrangling
numpy         Probability sampling
xgboost       Match outcome predictor
matplotlib    Visualizations
seaborn       Visualizations
tqdm          Progress bar for simulations
```

---

## Output

A full win probability distribution across all 48 teams, with stage-by-stage reach probabilities:

- Probability of reaching the Quarter-Finals
- Probability of reaching the Semi-Finals
- Probability of reaching the Final
- Probability of winning the tournament

---

## 📌 Notes

- 2026 uses a new **48-team / 12-group** format for the first time in World Cup history
- The top 2 from each group + the **8 best third-place teams** advance to the Round of 32
- All features use only pre-tournament data to avoid data leakage

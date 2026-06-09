# ⚽ FIFA World Cup 2026 — ML Winner Prediction

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
│   ├── raw/                              # Original Kaggle datasets, never modified
│   └── processed/                        # Cleaned and feature-engineered outputs from notebooks
│
├── notebooks/                            # End-to-end pipeline, run in order
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_simulation.ipynb
│   └── 05_streamlit_prep.ipynb
│
├── src/                                  # Reusable modules imported by notebooks and the app
│   ├── features.py
│   ├── model.py
│   └── simulation.py
│
├── app/                                  # Streamlit dashboard
│   └── app.py
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

**XGBoost** classifier trained on historical World Cup match data filtered to the 48 qualified 2026 nations (418 matches).

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
6. Third Place      → Simulate 1 match    → third place winner
7. Final            → Simulate 1 match    → tournament winner
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
streamlit     Interactive dashboard
```

---

## Output

A full win probability distribution across all 48 teams, with stage-by-stage reach probabilities:

- Probability of reaching the Quarter-Finals
- Probability of reaching the Semi-Finals
- Probability of reaching the Final
- Probability of winning the tournament

Results are visualized in an interactive **Streamlit dashboard** featuring:
- Full predicted tournament bracket with win probabilities
- Team-by-team profile and stage reach probabilities
- Monte Carlo insights (most common finals, biggest upsets)
- XGBoost feature importance

---

## Notes

- 2026 uses a new **48-team / 12-group** format for the first time in World Cup history
- The top 2 from each group + the **8 best third-place teams** advance to the Round of 32
- Training data filtered to 418 matches between currently qualified 2026 nations to ensure Elo feature coverage
- All features use only pre-tournament data to avoid data leakage

# NFL Game State Engine

An end-to-end NFL analytics project for 2025 season play-by-play data with:
- Win probability modeling
- QB efficiency analysis (EPA + CPOE)
- Fourth-down decision recommendations
- Monte Carlo game simulation
- Interactive Streamlit dashboard

## Project Scope

- Data source: `nfl_data_py`
- Season scope: **2025** (the 2025-26 NFL season window)
- Core cleaned dataset: `data/processed/pbp_clean.csv`

## Project Structure

```text
nfl-game-state-engine/
├── data/
│   ├── raw_pbp/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── win_probability_model.py
│   ├── qb_model.py
│   ├── fourth_down_engine.py
│   └── game_simulator.py
├── visualizations/
│   └── plots.py
└── dashboard/
    └── streamlit_app.py
```

## Setup

```bash
cd nfl-game-state-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install nfl_data_py pandas numpy scikit-learn xgboost matplotlib streamlit joblib
```

### macOS note (XGBoost)

If XGBoost fails with `libomp.dylib` errors:

```bash
brew install libomp
python -m pip install --upgrade --force-reinstall xgboost
```

## Run Pipeline

1. Build cleaned 2025 play-by-play data

```bash
python src/data_loader.py
```

2. Train win probability models (Logistic Regression + XGBoost)

```bash
python src/win_probability_model.py
```

3. Build QB efficiency dataset + scatter plot

```bash
python src/qb_model.py
```

4. Generate report visualizations

```bash
python visualizations/plots.py
```

5. Run dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Main Outputs

- Cleaned data: `data/processed/pbp_clean.csv`
- Win probability artifact: `models/win_probability.joblib`
- QB efficiency table: `data/processed/qb_efficiency.csv`
- Report charts: `visualizations/report/*.png`

## Dashboard Sections

- Overview
- QB EPA vs CPOE
- Fourth-Down Decision Engine
- Win Probability Timeline
- Monte Carlo Game Simulator

## Notes

- The fourth-down engine uses model-based expected win probability with scenario transitions for:
  - `GO_FOR_IT`
  - `PUNT`
  - `FIELD_GOAL`
- The game simulator runs drive-based Monte Carlo simulations and returns:
  - Win probability
  - Score distribution
  - Average score

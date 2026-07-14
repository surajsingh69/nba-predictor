# NBA Game Predictor

A  Python project that scrapes NBA data, engineers predictive features, and trains a machine learning model to predict game outcomes.

## Project pipeline

```
Raw scraping → API data → Feature engineering → ML model → Dashboard
```

## Folder structure

```
Project-Py/
├── data/
│   ├── raw_html/          # scraped HTML pages (mvp, player, team stats by year)
│   ├── nba_games/          # game-by-game scraper + parser + raw game data
│   │   ├── getdata.ipynb
│   │   ├── parse_data.ipynb
│   │   └── nba_games.csv   # one row per team-per-game, raw box scores
│   └── processed/          # cleaned, ready-to-use CSVs
│       ├── mvps.csv
│       ├── players.csv
│       ├── teams.csv
│       └── features.csv    # ML-ready feature table (output of build_features.py)
├── build_features.py       # turns nba_games.csv into features.csv
├── quick_signal_test.py    # sanity check: do the features actually predict anything?
├── test_predictions.py     # deeper validation: confusion matrix, ROC-AUC, calibration, spot-checks
├── webscraping.ipynb        # scrapes MVP / player / team season stats
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn
```

## How to run

**1. Build the feature table** (from raw game data → ML-ready features):
```bash
python build_features.py
```
Reads `data/nba_games/nba_games.csv`, outputs `data/processed/features.csv`.

**2. Sanity-check the features** (confirms they actually predict something before trusting them):
```bash
python quick_signal_test.py
```
Trains a simple logistic regression on a time-based train/test split and compares it against a "home team always wins" baseline. Current result: **66.8% accuracy vs. 52.8% baseline.**

**3. Deeper validation** (optional, but recommended before handoff):
```bash
python test_predictions.py
```
Goes beyond pass/fail — shows a confusion matrix, ROC-AUC (ranking quality), a calibration check (does "70% confident" really mean ~70% win rate?), and 5 example games with predicted vs. actual outcomes. Current result: **AUC 0.698**, well-calibrated across confidence buckets.

## What's in `features.csv`

6,922 rows (seasons 2021–2025), one row per team-per-game. Every column is a **pre-game** feature — nothing from the game's own result is used, to avoid data leakage.

| Column | Meaning |
|---|---|
| `team`, `team_opp` | team and opponent abbreviations |
| `date`, `season` | game date and season |
| `home` | 1 if this team was playing at home |
| `won` | target — did this team win? (True/False) |
| `*_roll10` | this team's rolling 10-game average of that stat (pts, fg%, ast, etc.), computed only from games before this one |
| `opp_*_roll10` | the opponent's equivalent rolling stats |
| `rest_days` | days since this team's previous game |
| `back_to_back` | 1 if playing on zero rest |
| `win_pct_last10` / `opp_win_pct_last10` | recent win percentage for each team |
| `h2h_win_pct` | this team's win rate specifically against this opponent, from past meetings |

## Team roles

| Person | Role | Status |
|---|---|---|
| Shakti | Scraping (game box scores) | ✅ Done |
| Asmit | Scraping (MVP/player/team stats) | ✅ Done |
| Suraj | Feature engineering | ✅ Done, validated |
| Samyog | Machine learning model | 🔲 In progress |
| — | API integration (live/upcoming games) | 🔲 Not started |
| — | Dashboard | 🔲 Not started |

## Notes for the ML teammate

- Each real game produces **2 rows** (one per team's perspective) — split train/test by `date`, not randomly, so both rows of a game stay together.
- A small number of `opp_*` columns have leftover NaNs (season-opener edge cases with no prior opponent history). `quick_signal_test.py` fills these with the train set's mean as a placeholder — worth deciding on a more deliberate approach.
- `teams.csv` (season-level team strength / SRS ratings) hasn't been merged in yet — could be a good next feature to add.
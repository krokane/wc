# WC 2026 Prediction Model

XGBoost Poisson regression model for predicting World Cup 2026 match outcomes. Trains on historical international football data (2003–2025) and generates win/draw/loss probabilities, expected goals, and scoreline distributions for upcoming fixtures. Includes a Dash dashboard for browsing predictions, tracking bets, and reviewing retroactive model accuracy.

---

## Setup

Requires Python ≥ 3.9 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv pip install -e .
```

Set the admin password before launching (defaults to `admin1234` if unset):

```bash
export WC_ADMIN_PASSWORD="yourpassword"
```

---

## CLI

```bash
uv run wc <command>
```

| Command | Description |
|---|---|
| `dash` | Launch the prediction dashboard (port 8050) |
| `scrape-elo` | Scrape Elo ratings and results → `data/elo_results.csv` |
| `process-stats` | Process FootyStats data → `data/fs_results.csv` |
| `build-features` | Build model feature matrix → `data/features.csv` |
| `train` | Train eval + deploy XGBoost models → `models/*.txt` |
| `evaluate` | Evaluate models on WC2026 holdout |
| `predict` | Predict upcoming fixtures with the deploy model |
| `save-predictions` | Snapshot today's predictions → `data/prediction_snapshots/` |
| `retro-predict` | Day-by-day retroactive predictions for past WC2026 games → `data/retro_predictions.csv` |

---

## How It Works

### Model

XGBoost with a Poisson objective (`count:poisson`) trained to predict goals scored per team-game. Features include Elo ratings, recent xG lags, historical goal statistics, and competition flags. Best iteration is determined once via early stopping on a held-out validation set (WC 2022 + continental 2024), then a full deploy model is retrained on all data.

Given predicted goals `λ_team` and `λ_opp`, win/draw/loss probabilities are derived from the Poisson joint distribution over the score grid.

### No-Leakage Retroactive Predictions

`retro-predict` retrains the model day-by-day for each WC2026 match date D, using only:
- All pre-2026 historical data
- WC2026 games already played **strictly before** date D (including their xG lags)

This mirrors what the model would have actually known on each match day — no data leakage from future results.

### xG Lags

After each WC2026 game, xG values are manually entered in the **xG Editor** tab. These flow into `features.py` so the next retrain incorporates up-to-date attacking/defensive form signals.

---

## Dashboard

```bash
uv run wc dash
```

### Predictions
Upcoming WC2026 fixtures with model win/draw/loss probabilities, expected goals, and implied American odds. Filterable by date. Click any match to open the analysis modal.

### Match Modal (4 tabs)
- **Betting Calculator** — enter market odds (American), see model edge and Kelly % for W/D/L and Draw No Bet markets. Log bets directly.
- **Total Goals** — over/under edge calculator for any line.
- **Scorelines** — top 8 most likely exact scores with probabilities and implied odds.
- **Parlay** — pick outcomes across same-day games; calculates combined probability, model-implied parlay odds, and edge vs. market parlay price.

### Bet Tracker
Log bets from the modal, mark results (won/lost/push), and track P&L and ROI.

### Pipeline
Run individual data pipeline stages or the full end-to-end pipeline from the UI.

### xG Editor
Enter post-match xG values for WC2026 games. Save, then run **Build Features** + **Train Model** to incorporate them.

### History
Past WC2026 results with retroactive model predictions. Shows overall accuracy and Ranked Probability Score (RPS). Click any past match to open the same analysis modal with the model's predictions from that day.

---

## Data Files

| File | Description |
|---|---|
| `data/elo_results.csv` | Match results + Elo ratings (scraped) |
| `data/fs_results.csv` | FootyStats per-game xG and shot stats |
| `data/tm_results.csv` | Transfermarkt squad value data |
| `data/features.csv` | Full model feature matrix |
| `data/manual_xg.csv` | Manually entered WC2026 xG values |
| `data/retro_predictions.csv` | Day-by-day retroactive WC2026 predictions |
| `data/prediction_snapshots/` | Daily snapshots of upcoming predictions |
| `models/xgb_goals.txt` | Eval model (trained without WC2026 data) |
| `models/xgb_goals_deploy.txt` | Deploy model (trained on all data) |

# predictor_ML_model

The machine learning / pattern-recognition track for Group 21's Stock Trend Predictor
(BIT 268, Pattern Recognition pillar). Trains a per-stock classifier (logistic regression
or random forest, whichever tests better) to predict next-day price direction for 50
large-cap US equities, and sends the results to the backend as a nightly batch job.

This repo does NOT include the frontend or backend - see the other repos in this org:
- Frontend: (link to your frontend repo)
- Backend: `BIT268-group21/predictorBackend`

## Quick start

```bash
pip install -r requirements.txt
export TWELVE_DATA_API_KEY=your_key_here
python scripts/pull_and_check.py       # pulls fresh price data for all 50 tickers
python scripts/run_nightly_batch.py    # trains, evaluates, predicts, and sends (or dry-runs)
```

See `DEPLOYMENT.md` for how this runs in production.

## Pipeline overview

**Production** (`scripts/`) - everything the nightly job actually needs to run, verified
self-contained (no dependency on anything in `scripts/analysis/`):

| Script | Purpose |
|---|---|
| `pull_and_check.py` | Pulls fresh OHLCV data for all 50 tickers from Twelve Data |
| `features.py` | Computes the 11 candidate technical-indicator features per stock |
| `selection.py` | Per-stock top-5 feature selection by correlation with target |
| `training.py` | Trains + evaluates both logistic regression and random forest per stock |
| `build_payload.py` | Retrain-on-full-history + predict logic, imported by run_nightly_batch.py |
| `run_nightly_batch.py` | **The actual production entry point** - wraps everything above, retrains fresh every run, sends the batch (or dry-runs if no endpoint is configured) |

**Analysis / reporting tools** (`scripts/analysis/`) - not required for production, but this
is how the pipeline was built and validated step-by-step, and how the results in the project's
decisions log were generated:

| Script | Purpose |
|---|---|
| `sectors.py` | Sector classification for the 50 tickers |
| `run_eda.py` | Exploratory data analysis: class balance, volatility, sector patterns |
| `run_features.py` | Regenerates `data/featured/` for manual inspection |
| `run_training.py` | One-time training run + evaluation (used to validate the approach) |
| `select_winner.py` | One-time model selection (used to validate the approach) |

For the full methodology, honest results, and all project decisions, see the project's
central decisions log (linked internally within the team - ask a teammate if you don't
have access).

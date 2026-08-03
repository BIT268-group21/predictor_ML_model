# Deployment Guide - ML Batch Job

This describes how to run and deploy the ML prediction pipeline. It does NOT cover
deploying the frontend or backend - see their respective repos for that.

## What this service does

Once daily, it:
1. Pulls fresh price data for 50 stocks
2. Re-engineers features and re-selects the best 5 per stock
3. Retrains both a logistic regression and a random forest per stock (chronological
   80/20 split, so accuracy is honestly measured on unseen data every single run)
4. Picks whichever model tested better, retrains it on ALL available history, and
   predicts tomorrow's direction
5. Sends all 50 predictions as one batch POST to the backend

The model is NOT trained once and reused - it retrains from scratch every run. This is
deliberate (see the project decisions log, "nightly retraining"): it keeps the reported
accuracy honest rather than stale, and the pipeline is cheap enough (well under a minute
for all 50 stocks) that there's no real cost to doing it properly every time.

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt` (pandas, numpy, scikit-learn, requests)
- A Twelve Data API key (free tier is sufficient)

## Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `TWELVE_DATA_API_KEY` | Yes | Used by `pull_and_check.py` to fetch price data |
| `BATCH_ENDPOINT_URL` | No | Full URL of the backend's `POST /api/predictions/batch`. **If unset, the job automatically runs in dry-run mode** - it still builds the full payload and saves it to `data/final_payload.json`, but sends nothing over the network. This is the safe default. |
| `BATCH_TIMEOUT_SECONDS` | No | HTTP timeout for the POST call. Defaults to 30. |

**On Render specifically:** environment variables are NOT read from a committed `.env`
file - set them via the service's Environment tab in the Render dashboard (or in
`render.yaml` if using Blueprints). A local `.env` file works fine for your own machine
during development; Render just doesn't read one from the repo at runtime.

## Running locally

```bash
pip install -r requirements.txt
export TWELVE_DATA_API_KEY=your_key_here
python scripts/pull_and_check.py
python scripts/run_nightly_batch.py
```

With `BATCH_ENDPOINT_URL` unset, this is fully safe to run repeatedly - it never sends
anything anywhere, just builds and saves the payload locally so you can inspect it.

## Deploying as a Render Cron Job

1. Create a new **Cron Job** service (not a Web Service) in Render, pointed at this repo
2. Build command: `pip install -r requirements.txt`
3. Command to run on schedule:
   ```
   python scripts/pull_and_check.py && python scripts/run_nightly_batch.py
   ```
4. Set environment variables in the Render dashboard: `TWELVE_DATA_API_KEY` and
   `BATCH_ENDPOINT_URL` (the backend's real deployed URL, once it exists)
5. Set the schedule - remember Render's cron schedule is evaluated in UTC, so convert
   whatever local "midnight" you actually want
6. **Networking:** this service and the backend need to be able to reach each other over
   Render's private network, since the backend endpoint has no authentication by design
   (it's meant to only be reachable from this job, not the public internet). Confirm with
   whoever manages the backend's Render service that this is set up before relying on it.

## Troubleshooting

- **"Missing raw data for: [...]"** - `pull_and_check.py` didn't run first, or failed
  partway. Check that `TWELVE_DATA_API_KEY` is set and valid.
- **Payload built but nothing sent** - check that `BATCH_ENDPOINT_URL` is actually set in
  the environment. If it's empty/unset, dry-run is the expected, safe behavior.
- **Connection refused / timeout on POST** - the backend isn't reachable at that URL.
  Check the private networking setup between this service and the backend on Render.

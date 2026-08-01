"""
THE NIGHTLY PRODUCTION BATCH JOB.

Wraps the full pipeline end-to-end, per decisions.md Section 2d (retrain every
night, not train-once-and-freeze):

    engineer_features()  -> per-stock feature engineering
    select_features()    -> per-stock top-5 feature selection
    train_and_evaluate()  -> chronological 80/20 split, fresh honest model_accuracy
    train_final_model()  -> retrain winner on ALL history, predict on latest row
    -> assemble payload matching decisions.md Section 3a
    -> POST to the backend, or DRY RUN if no endpoint URL is configured

This is meant to be the actual command a Render Cron Job runs nightly (after
scripts/pull_and_check.py has already refreshed data/raw/ - see the note below).

Usage:
    python scripts/run_nightly_batch.py

Configuration (environment variables - set via Render's dashboard in production,
a local .env file in development; Render does not read a committed .env file
at runtime, see decisions.md Section 2d):

    BATCH_ENDPOINT_URL     - full URL of POST /api/predictions/batch.
                              If unset/empty, automatically falls back to DRY RUN:
                              the payload is still built and saved locally, but
                              nothing is sent over the network. This is the
                              expected state until decisions.md Section 10's
                              production URL is resolved.
    BATCH_TIMEOUT_SECONDS  - optional, defaults to 30

NOTE ON DATA FRESHNESS: this script does NOT re-pull price data itself - that's
scripts/pull_and_check.py's job. In the Render Cron Job, chain both in sequence,
e.g.: `python scripts/pull_and_check.py && python scripts/run_nightly_batch.py`
"""
import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features
from selection import select_features
from training import train_and_evaluate
from build_payload import train_final_model, next_trading_day

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX",
    "KO", "PEP", "ABBV", "COST", "MRK", "ADBE", "CRM", "NFLX", "INTC", "AMD",
    "CSCO", "PFE", "TMO", "ABT", "NKE", "MCD", "ORCL", "IBM", "QCOM", "TXN",
    "HON", "UPS", "CAT", "GS", "MS", "BA", "GE", "LMT", "SBUX", "LOW",
]

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/final_payload.json"


def pick_winner(lr_acc, rf_acc):
    """Strictly higher accuracy wins, no threshold (decisions.md Section 9/14).
    Exact ties default to LR - simpler, easier for the LLM explanation step
    to translate its coefficients into plain language (Section 14)."""
    if rf_acc > lr_acc:
        return 'RF'
    return 'LR'


def build_nightly_payload():
    import pandas as pd

    predictions = []
    skipped = []

    for ticker in TICKERS:
        fname = os.path.join(RAW_DIR, f"{ticker.replace('.', '_')}.csv")
        if not os.path.exists(fname):
            skipped.append(ticker)
            continue

        df = pd.read_csv(fname)
        featured = engineer_features(df)
        selected, _ = select_features(featured)

        # Fresh honest accuracy every night: chronological 80/20 split (Section 2d)
        eval_result = train_and_evaluate(featured, selected)
        lr_acc = eval_result['logistic_regression']['test_accuracy']
        rf_acc = eval_result['random_forest']['test_accuracy']
        algorithm = pick_winner(lr_acc, rf_acc)
        model_accuracy = max(lr_acc, rf_acc)

        # Retrain on ALL history for the actual live prediction (build_payload.py's approach)
        pred_class, confidence, latest_row = train_final_model(featured, selected, algorithm)

        prediction_date = latest_row['datetime'].values[0]
        target_date = next_trading_day(prediction_date)
        features_dict = {feat: float(latest_row[feat].values[0]) for feat in selected}

        predictions.append({
            "ticker": ticker,
            "prediction_date": prediction_date,
            "target_date": target_date,
            "predicted_direction": "up" if pred_class == 1 else "down",
            "confidence": round(confidence, 4),
            "model_accuracy": round(model_accuracy, 4),
            "features": features_dict,
            "last_close_price": float(latest_row['close'].values[0]),
        })

    if skipped:
        print(f"WARNING: skipped {len(skipped)} tickers with missing raw data: {skipped}")

    return {"predictions": predictions}


def send_or_dry_run(payload):
    url = os.environ.get('BATCH_ENDPOINT_URL', '').strip()
    timeout = int(os.environ.get('BATCH_TIMEOUT_SECONDS', '30'))

    if not payload['predictions']:
        raise RuntimeError(
            "No predictions were assembled (0/50 tickers). This almost always means "
            "data/raw/ is missing or empty - did scripts/pull_and_check.py run first? "
            "Failing loudly here on purpose, so a misconfigured cron job shows as FAILED "
            "instead of silently sending an empty batch."
        )

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} ({len(payload['predictions'])} predictions)")

    if not url:
        print("BATCH_ENDPOINT_URL not set -> DRY RUN. Payload built and saved locally, nothing sent over the network.")
        return

    import requests
    print(f"Posting batch to {url} ...")
    resp = requests.post(url, json=payload, timeout=timeout)
    print(f"Response: {resp.status_code} {resp.text[:500]}")
    resp.raise_for_status()


def main():
    print(f"Nightly batch job starting at {datetime.now(timezone.utc).isoformat()}")
    payload = build_nightly_payload()
    up = sum(1 for p in payload['predictions'] if p['predicted_direction'] == 'up')
    total = len(payload['predictions'])
    print(f"Assembled {total} predictions ({up} up, {total - up} down)")
    send_or_dry_run(payload)
    print("Nightly batch job complete.")


if __name__ == "__main__":
    main()

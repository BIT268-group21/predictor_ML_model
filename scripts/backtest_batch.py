"""
scripts/backtest_batch.py

*** THIS PRODUCES BACKTESTED PREDICTIONS, NOT LIVE ONES. ***
Every prediction below comes from a FRESH RETRAIN of the full pipeline
(engineer_features -> select_features -> train_and_evaluate -> train_final_model)
run on data truncated to a historical cutoff date, walking backwards over the
last N trading days, then compared against real, already-known outcomes.
This is NOT the organically-accumulated stream of live predictions that
scripts/run_nightly_batch.py posts one night at a time - do not confuse the
two, and do not treat this script's output as a substitute for letting the
real nightly job accumulate its own history over time.

WHAT THIS DOES
    For each ticker in run_nightly_batch.TICKERS:
      1. Load data/raw/{ticker}.csv (the same file the nightly job reads).
         This script does NOT re-pull from Twelve Data - if data/raw looks
         stale or missing, run scripts/pull_and_check.py first.
      2. Determine the last --days trading days present in that ticker's CSV,
         EXCLUDING the most recent row. That most recent date is "today" for
         run_nightly_batch.py's next live run, so it's left for the live job
         rather than duplicated here.
      3. For each of those cutoff dates, oldest to newest:
           a. Slice the raw DataFrame to rows with datetime <= cutoff BEFORE
              calling engineer_features(). This is the crux of a walk-forward
              backtest: no feature or model at a given cutoff may ever see a
              row dated after that cutoff (no lookahead leakage).
           b. Run engineer_features() -> select_features() -> train_and_evaluate()
              on the slice for a fresh, honest model_accuracy at that point in
              history (same pattern as run_nightly_batch.build_nightly_payload()).
           c. Run train_final_model() on the slice to get the predicted
              direction/confidence for the slice's last row (i.e. "today" as
              of that historical cutoff).
           d. Assemble one prediction object, identical in shape to
              build_nightly_payload()'s output (ticker, prediction_date,
              target_date, predicted_direction, confidence, model_accuracy,
              features, last_close_price).
    Predictions are grouped into one payload per distinct prediction_date
    (not per ticker, not all merged together) - each historical day is its
    own nightly-shaped batch across all tickers, POSTed (or dry-run-saved)
    in chronological order, oldest first. If any single day's POST fails,
    it's logged and the run continues; failures are summarized at the end.

    Note on cost: this reruns the equivalent of one full nightly batch job
    --days times (default 30), so expect roughly 30x run_nightly_batch.py's
    runtime.

USAGE
    python scripts/backtest_batch.py [--days 30]

CONFIGURATION (same variables/behavior as run_nightly_batch.py)
    BATCH_ENDPOINT_URL     - full URL of POST /api/predictions/batch.
                              If unset/empty: DRY RUN - each day's payload is
                              written to data/backtest_payloads/ instead of
                              being sent over the network.
    BATCH_AUTH_TOKEN       - bearer token; only read/required when actually
                              POSTing (BATCH_ENDPOINT_URL set).
    BATCH_TIMEOUT_SECONDS  - optional, defaults to 30

GRADING NOTE: posting these payloads does not grade them. Grading happens
later, on the backend's own AccuracyCheckJob (nightly at 19:00 UTC), which
compares each prediction's predicted_direction against the real close on its
target_date.
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import requests

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features
from selection import select_features
from training import train_and_evaluate
from build_payload import train_final_model, next_trading_day
from run_nightly_batch import TICKERS, RAW_DIR, pick_winner

OUTPUT_DIR = "data/backtest_payloads"
DEFAULT_DAYS = 30


def build_backtest_predictions(days=DEFAULT_DAYS):
    """
    Walks the last `days` trading days (excluding each ticker's most recent
    row - see module docstring) and returns predictions grouped by
    prediction_date: {prediction_date: [prediction, ...], ...}.

    Reuses engineer_features() / select_features() / train_and_evaluate() /
    train_final_model() exactly as run_nightly_batch.build_nightly_payload()
    does - the only difference is that each cutoff date gets its own
    backward-truncated slice of the raw DataFrame instead of the full history.
    """
    import pandas as pd

    predictions_by_date = defaultdict(list)
    skipped = []

    for ticker in TICKERS:
        fname = os.path.join(RAW_DIR, f"{ticker.replace('.', '_')}.csv")
        if not os.path.exists(fname):
            skipped.append(ticker)
            continue

        df = pd.read_csv(fname)  # oldest-first, per pull_and_check.py

        # Last `days` trading days present, EXCLUDING the most recent row -
        # that date belongs to run_nightly_batch.py's live "today", not this
        # backtest (see module docstring).
        available = df['datetime'].tolist()[:-1]
        cutoff_dates = available[-days:]

        if len(cutoff_dates) < days:
            print(f"WARNING: {ticker} only has {len(cutoff_dates)} eligible "
                  f"historical trading day(s) (< {days} requested) - backtesting all of them.")

        for cutoff_date in cutoff_dates:
            # Slice BEFORE feature engineering - no row dated after cutoff_date
            # may influence features/training/prediction at this cutoff.
            slice_df = df[df['datetime'] <= cutoff_date].reset_index(drop=True)

            featured = engineer_features(slice_df)
            selected, _ = select_features(featured)

            # Fresh honest accuracy for this historical point in time (same
            # chronological 80/20 split build_nightly_payload() uses nightly).
            eval_result = train_and_evaluate(featured, selected)
            lr_acc = eval_result['logistic_regression']['test_accuracy']
            rf_acc = eval_result['random_forest']['test_accuracy']
            algorithm = pick_winner(lr_acc, rf_acc)
            model_accuracy = max(lr_acc, rf_acc)

            # Retrain on all history up to this cutoff for the actual
            # historical prediction (build_payload.py's approach).
            pred_class, confidence, latest_row = train_final_model(featured, selected, algorithm)

            prediction_date = latest_row['datetime'].values[0]
            target_date = next_trading_day(prediction_date)
            features_dict = {feat: float(latest_row[feat].values[0]) for feat in selected}

            predictions_by_date[prediction_date].append({
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

    return predictions_by_date


def send_or_dry_run_day(prediction_date, payload):
    """Mirrors run_nightly_batch.py's send_or_dry_run() auth/dry-run pattern,
    one historical day's payload at a time.

    Returns (ok: bool, error: str | None).
    """
    url = os.environ.get('BATCH_ENDPOINT_URL', '').strip()
    timeout = int(os.environ.get('BATCH_TIMEOUT_SECONDS', '30'))

    if not url:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{prediction_date}.json")
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[{prediction_date}] DRY RUN -> wrote {out_path} ({len(payload['predictions'])} predictions)")
        return True, None

    headers = {"Authorization": f"Bearer {os.environ['BATCH_AUTH_TOKEN']}"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        status = getattr(exc.response, 'status_code', None)
        body = getattr(exc.response, 'text', str(exc))[:200] if exc.response is not None else str(exc)
        print(f"[{prediction_date}] FAILED - status={status} {body}")
        return False, f"status={status} {body}"

    # Best-effort extraction of a "saved" count from the response body -
    # falls back to the number sent if the backend's response shape doesn't
    # include one we recognize.
    saved_count = len(payload['predictions'])
    try:
        body = resp.json()
        if isinstance(body, dict):
            saved_count = body.get('saved', body.get('count', saved_count))
    except ValueError:
        pass

    print(f"[{prediction_date}] {resp.status_code} - saved {saved_count} "
          f"(sent {len(payload['predictions'])}): {resp.text[:200]}")
    return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of the nightly batch pipeline over historical trading days."
    )
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS,
                         help=f"number of historical trading days to backtest (default: {DEFAULT_DAYS})")
    args = parser.parse_args()

    print(f"Backtest batch job starting at {datetime.now(timezone.utc).isoformat()} "
          f"({args.days} trading days)")

    predictions_by_date = build_backtest_predictions(days=args.days)

    if not predictions_by_date:
        raise RuntimeError(
            "No predictions were assembled for any historical day. This almost always means "
            "data/raw/ is missing or empty - did scripts/pull_and_check.py run first?"
        )

    ordered_dates = sorted(predictions_by_date.keys())
    print(f"Assembled predictions for {len(ordered_dates)} historical day(s): "
          f"{ordered_dates[0]} .. {ordered_dates[-1]}")

    succeeded = []
    failed = []
    total_posted = 0

    for prediction_date in ordered_dates:
        preds = predictions_by_date[prediction_date]
        payload = {"predictions": preds}
        ok, err = send_or_dry_run_day(prediction_date, payload)
        if ok:
            succeeded.append(prediction_date)
            total_posted += len(preds)
        else:
            failed.append((prediction_date, err))

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Days succeeded: {len(succeeded)}/{len(ordered_dates)}")
    print(f"Days failed:    {len(failed)}/{len(ordered_dates)}")
    if failed:
        for d, err in failed:
            print(f"  - {d}: {err}")
    print(f"Total predictions posted/saved: {total_posted}")
    print(
        "\nNOTE: this script only assembles and sends predictions - it does NOT grade them. "
        "Grading happens on the backend's own AccuracyCheckJob (nightly at 19:00 UTC), which "
        "compares each prediction's predicted_direction against the real close on target_date."
    )


if __name__ == "__main__":
    main()

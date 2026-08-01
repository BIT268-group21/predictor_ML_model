"""
Trains both logistic regression and random forest for every stock in
data/raw/, using each stock's own selected features (from selection.py).

Usage:
    python scripts/run_training.py

Outputs (not committed - derived data, regenerate on demand):
    data/training_summary.csv    - one row per stock: accuracies, baseline, which model beat baseline
    data/training_results.pkl    - full trained models + predictions for all 50 stocks (used in Step 5 evaluation)

NOTE: training_results.pkl will be several MB (pickled sklearn models x 2 x 50 stocks).
That's expected and why it's gitignored rather than committed.
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features
from selection import select_features
from training import train_and_evaluate

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX",
    "KO", "PEP", "ABBV", "COST", "MRK", "ADBE", "CRM", "NFLX", "INTC", "AMD",
    "CSCO", "PFE", "TMO", "ABT", "NKE", "MCD", "ORCL", "IBM", "QCOM", "TXN",
    "HON", "UPS", "CAT", "GS", "MS", "BA", "GE", "LMT", "SBUX", "LOW",
]

RAW_DIR = "data/raw"


def main():
    import pandas as pd

    all_results = {}
    rows = []
    missing = []

    for ticker in TICKERS:
        fname = os.path.join(RAW_DIR, f"{ticker.replace('.', '_')}.csv")
        if not os.path.exists(fname):
            missing.append(ticker)
            continue

        df = pd.read_csv(fname)
        featured = engineer_features(df)
        selected, _ = select_features(featured)
        result = train_and_evaluate(featured, selected)
        all_results[ticker] = result

        y_test = result['y_test']
        naive_baseline = max(y_test.mean(), 1 - y_test.mean())
        lr_acc = result['logistic_regression']['test_accuracy']
        rf_acc = result['random_forest']['test_accuracy']

        rows.append({
            'ticker': ticker,
            'selected_features': ','.join(selected),
            'n_train': result['n_train'],
            'n_test': result['n_test'],
            'test_majority_baseline': naive_baseline,
            'lr_test_accuracy': lr_acc,
            'rf_test_accuracy': rf_acc,
            'lr_beats_baseline': lr_acc > naive_baseline,
            'rf_beats_baseline': rf_acc > naive_baseline,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv('data/training_summary.csv', index=False)

    with open('data/training_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Trained both models for {len(all_results)}/{len(TICKERS)} stocks")
    if missing:
        print(f"Missing raw data for: {missing} (run scripts/pull_and_check.py first)")
    print(f"Wrote data/training_summary.csv and data/training_results.pkl")
    print(f"\nLR beats test-set majority baseline: {summary['lr_beats_baseline'].sum()}/{len(summary)} stocks")
    print(f"RF beats test-set majority baseline: {summary['rf_beats_baseline'].sum()}/{len(summary)} stocks")


if __name__ == "__main__":
    main()

"""
Regenerates data/featured/*.csv from data/raw/*.csv using engineer_features().

Usage:
    python scripts/run_features.py

Cheap and fast to rerun (no API calls) - data/featured/ is derived data,
so it's intentionally NOT committed to the repo. Re-run this any time
data/raw/ changes, or after a fresh pull_and_check.py run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX",
    "KO", "PEP", "ABBV", "COST", "MRK", "ADBE", "CRM", "NFLX", "INTC", "AMD",
    "CSCO", "PFE", "TMO", "ABT", "NKE", "MCD", "ORCL", "IBM", "QCOM", "TXN",
    "HON", "UPS", "CAT", "GS", "MS", "BA", "GE", "LMT", "SBUX", "LOW",
]

RAW_DIR = "data/raw"
FEATURED_DIR = "data/featured"


def main():
    import pandas as pd  # local import so the script fails fast with a clear error if pandas is missing

    os.makedirs(FEATURED_DIR, exist_ok=True)
    saved, missing = 0, []

    for ticker in TICKERS:
        fname = os.path.join(RAW_DIR, f"{ticker.replace('.', '_')}.csv")
        if not os.path.exists(fname):
            missing.append(ticker)
            continue
        df = pd.read_csv(fname)
        featured = engineer_features(df)
        out_path = os.path.join(FEATURED_DIR, f"{ticker.replace('.', '_')}.csv")
        featured.to_csv(out_path, index=False)
        saved += 1

    print(f"Saved {saved}/{len(TICKERS)} featured CSVs to {FEATURED_DIR}/")
    if missing:
        print(f"Missing raw data for: {missing} (run scripts/pull_and_check.py first)")


if __name__ == "__main__":
    main()

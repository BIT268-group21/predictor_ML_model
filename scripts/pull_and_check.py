import requests
import time
import csv
import os
from datetime import datetime

API_KEY = os.environ["TWELVE_DATA_API_KEY"]  # set locally via .env, or as a GitHub Actions secret
BASE_URL = "https://api.twelvedata.com/time_series"
OUTPUT_DIR = "data/raw"
DELAY_SECONDS = 8  # stay safely under 8 req/min

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX",
    "KO", "PEP", "ABBV", "COST", "MRK", "ADBE", "CRM", "NFLX", "INTC", "AMD",
    "CSCO", "PFE", "TMO", "ABT", "NKE", "MCD", "ORCL", "IBM", "QCOM", "TXN",
    "HON", "UPS", "CAT", "GS", "MS", "BA", "GE", "LMT", "SBUX", "LOW",
]

# Fallback symbol formats to try if the primary one fails (mainly for BRK.B-style tickers)
FALLBACKS = {
    "BRK.B": ["BRK.B", "BRK-B", "BRKB"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_symbol(symbol):
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 800,  # comfortably covers ~3 years of trading days
        "apikey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    try:
        data = resp.json()
    except Exception:
        return None, f"non-JSON response (HTTP {resp.status_code})"

    if data.get("status") == "error" or "values" not in data:
        return None, data.get("message", "unknown error / no values field")

    return data, None


def check_gaps(dates_sorted):
    """Rough gap check: flag any consecutive weekday-to-weekday gap > 4 calendar days
    (catches missing trading days beyond normal weekends; doesn't try to model holidays)."""
    gaps = []
    for i in range(1, len(dates_sorted)):
        prev = datetime.strptime(dates_sorted[i - 1], "%Y-%m-%d")
        curr = datetime.strptime(dates_sorted[i], "%Y-%m-%d")
        delta_days = (curr - prev).days
        if delta_days > 4:  # more than a long weekend
            gaps.append((dates_sorted[i - 1], dates_sorted[i], delta_days))
    return gaps


results = []

for i, ticker in enumerate(TICKERS):
    symbol_attempts = FALLBACKS.get(ticker, [ticker])
    data = None
    used_symbol = None
    error_msg = None

    for attempt_symbol in symbol_attempts:
        data, error_msg = fetch_symbol(attempt_symbol)
        if data is not None:
            used_symbol = attempt_symbol
            break
        time.sleep(DELAY_SECONDS)  # still respect rate limit even on failed attempts

    if data is None:
        results.append({
            "ticker": ticker,
            "status": "FAILED",
            "used_symbol": None,
            "rows": 0,
            "earliest": None,
            "latest": None,
            "gap_count": None,
            "error": error_msg,
        })
        print(f"[{i+1}/50] {ticker}: FAILED ({error_msg})")
        time.sleep(DELAY_SECONDS)
        continue

    values = data["values"]  # newest first, per Twelve Data convention
    dates_sorted = sorted(v["datetime"] for v in values)
    rows = len(values)
    earliest, latest = dates_sorted[0], dates_sorted[-1]
    gaps = check_gaps(dates_sorted)

    # Save raw data, oldest first for downstream convenience
    values_sorted = sorted(values, key=lambda v: v["datetime"])
    out_path = os.path.join(OUTPUT_DIR, f"{ticker.replace('.', '_')}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for row in values_sorted:
            writer.writerow(row)

    status = "OK"
    if rows < 500:
        status = "SHORT_HISTORY"
    elif gaps:
        status = "HAS_GAPS"

    results.append({
        "ticker": ticker,
        "status": status,
        "used_symbol": used_symbol,
        "rows": rows,
        "earliest": earliest,
        "latest": latest,
        "gap_count": len(gaps),
        "error": None,
    })

    flag = "" if used_symbol == ticker else f" (used symbol '{used_symbol}')"
    print(f"[{i+1}/50] {ticker}: {status} — {rows} rows, {earliest} to {latest}, {len(gaps)} gap(s){flag}")

    time.sleep(DELAY_SECONDS)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

ok = [r for r in results if r["status"] == "OK"]
short = [r for r in results if r["status"] == "SHORT_HISTORY"]
gappy = [r for r in results if r["status"] == "HAS_GAPS"]
failed = [r for r in results if r["status"] == "FAILED"]

print(f"Clean (OK): {len(ok)}/50")
print(f"Short history (<500 rows): {len(short)}/50 -> {[r['ticker'] for r in short]}")
print(f"Has gaps (>4-day jumps): {len(gappy)}/50 -> {[r['ticker'] for r in gappy]}")
print(f"Failed to fetch: {len(failed)}/50 -> {[(r['ticker'], r['error']) for r in failed]}")

# Save full summary to CSV for reference
summary_path = "data/pull_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ticker", "status", "used_symbol", "rows", "earliest", "latest", "gap_count", "error"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\nFull summary saved to {summary_path}")
print(f"Raw per-ticker CSVs saved to {OUTPUT_DIR}/")

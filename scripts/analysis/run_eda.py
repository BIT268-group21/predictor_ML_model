"""
Computes per-stock and per-sector EDA statistics (class balance, return/volatility)
from data/raw/*.csv, using engineer_features() for the target and daily_return columns.

Usage:
    python scripts/run_eda.py

Outputs (not committed - derived data, regenerate on demand):
    data/eda_summary.csv         - one row per stock
    data/eda_sector_summary.csv  - one row per sector (aggregated)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features
from sectors import SECTOR_MAP

RAW_DIR = "data/raw"


def main():
    import pandas as pd

    rows = []
    for ticker, sector in SECTOR_MAP.items():
        fname = os.path.join(RAW_DIR, f"{ticker.replace('.', '_')}.csv")
        if not os.path.exists(fname):
            print(f"Skipping {ticker}: raw data missing")
            continue
        df = pd.read_csv(fname)
        featured = engineer_features(df)
        clean = featured.dropna(subset=['target'])
        daily_ret = clean['daily_return'].dropna()

        pct_up = clean['target'].mean()
        rows.append({
            'ticker': ticker,
            'sector': sector,
            'pct_up_days': pct_up,
            'pct_down_days': 1 - pct_up,
            'mean_daily_return': daily_ret.mean(),
            'std_daily_return': daily_ret.std(),
            'min_close': clean['close'].min(),
            'max_close': clean['close'].max(),
            'total_return_pct': (clean['close'].iloc[-1] / clean['close'].iloc[0] - 1) * 100,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv('data/eda_summary.csv', index=False)

    sector_summary = summary.groupby('sector').agg(
        n_stocks=('ticker', 'count'),
        avg_pct_up=('pct_up_days', 'mean'),
        avg_volatility=('std_daily_return', 'mean'),
        avg_total_return=('total_return_pct', 'mean'),
    ).sort_values('avg_volatility', ascending=False)
    sector_summary.to_csv('data/eda_sector_summary.csv')

    print(f"Wrote data/eda_summary.csv ({len(summary)} stocks)")
    print(f"Wrote data/eda_sector_summary.csv ({len(sector_summary)} sectors)")


if __name__ == "__main__":
    main()

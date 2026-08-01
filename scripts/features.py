import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Takes a raw OHLCV DataFrame for ONE stock (columns: datetime, open, high, low, close, volume)
    and returns a new DataFrame with all candidate features + target added.
    Assumes df is already sorted oldest-to-newest.
    """
    df = df.copy()  # never mutate the caller's original DataFrame

    # --- price-level features (from the mini-project) ---
    df['ma_7'] = df['close'].rolling(7).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    df['ma_diff'] = df['ma_7'] - df['ma_30']

    # --- returns & volatility ---
    df['daily_return'] = df['close'].pct_change()
    df['volatility_7'] = df['daily_return'].rolling(7).std()

    # --- momentum / lagged returns (from Extended Experiments) ---
    df['lag_return_1'] = df['daily_return'].shift(1)
    df['lag_return_2'] = df['daily_return'].shift(2)
    df['lag_return_3'] = df['daily_return'].shift(3)
    df['momentum_5'] = df['close'].pct_change(5)

    # --- volume & price-vs-MA ---
    df['volume_change'] = df['volume'].pct_change()
    df['price_vs_ma7'] = (df['close'] - df['ma_7']) / df['ma_7']
    df['price_vs_ma30'] = (df['close'] - df['ma_30']) / df['ma_30']

    # --- target: did price go up the next day? ---
    # NOTE: comparing NaN > x in pandas silently returns False, not NaN — casting
    # straight to int would turn the last row's "unknown" into a fake "down" label.
    # So: compute as float, let the NaN survive, and leave it for the NaN-cleanup
    # step to drop later (same principle as the mini-project's Step 4).
    next_close = df['close'].shift(-1)
    df['target'] = np.where(next_close.isna(), np.nan, (next_close > df['close']).astype(float))

    return df


# The 11 candidate features (excludes datetime/OHLCV raw columns, the intermediate
# daily_return column, and target)
CANDIDATE_FEATURES = [
    'ma_7', 'ma_30', 'ma_diff',
    'volatility_7',
    'lag_return_1', 'lag_return_2', 'lag_return_3', 'momentum_5',
    'volume_change',
    'price_vs_ma7', 'price_vs_ma30',
]

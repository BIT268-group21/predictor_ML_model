"""
Assembles the final batch prediction payload for all 50 stocks, matching the
Data Contract in decisions.md Section 3a exactly.

For each stock:
  1. Load the winning algorithm (LR or RF) from data/final_model_selection.csv
  2. Retrain that algorithm on ALL available history (not just the 80% train split -
     the held-out test accuracy is still what gets reported as model_accuracy, but
     the live prediction itself should use every row of real data available)
  3. Predict on the most recent row (valid features, unknown target - the row that
     was previously unusable because of the NaN target)
  4. Assemble one prediction object per stock

Usage:
    python scripts/build_payload.py

Output:
    data/final_payload.json  - the assembled batch payload (dry-run only, never POSTed -
                                no reachable backend endpoint exists yet per decisions.md
                                Section 10/13 - see scripts/send_batch.py for the real sender)

Reusable entry point: assemble_payload() - imported directly by scripts/send_batch.py.
"""
import os
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features
from selection import select_features

RAW_DIR = "data/raw"
SELECTION_CSV = "data/final_model_selection.csv"


def next_trading_day(date_str):
    """Naive next-weekday calculation - skips weekends, does NOT account for
    market holidays. Known limitation, acceptable for this project's scope."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    d += timedelta(days=1)
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def train_final_model(featured_df, selected_features, algorithm):
    """Retrains the winning algorithm on ALL available history (train+test combined),
    then predicts on the most recent row (valid features, NaN target)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # rows usable for training: everything with complete features AND a known target
    history = featured_df.dropna(subset=selected_features + ['target'])
    X_hist = history[selected_features]
    y_hist = history['target'].astype(int)

    # the live prediction row: most recent row, valid features, target unknown (NaN)
    latest_row = featured_df.iloc[[-1]]
    X_latest = latest_row[selected_features]

    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(X_hist)
    X_latest_scaled = scaler.transform(X_latest)

    if algorithm == 'RF':
        model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42)
    else:  # 'LR' or 'Tie' - default to LR since it's simpler (see decisions.md discussion)
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_hist_scaled, y_hist)
    pred_class = int(model.predict(X_latest_scaled)[0])
    pred_proba = model.predict_proba(X_latest_scaled)[0]
    confidence = float(pred_proba[pred_class])

    return pred_class, confidence, latest_row


def assemble_payload(selection_csv=SELECTION_CSV, raw_dir=RAW_DIR):
    """
    Builds the full {"predictions": [...]} payload for all 50 stocks.
    Reusable entry point - imported directly by scripts/send_batch.py so the
    real POST logic never has to duplicate this assembly code.
    """
    import pandas as pd

    selection = pd.read_csv(selection_csv)
    selection = selection.set_index('ticker')

    predictions = []

    for ticker in selection.index:
        fname = os.path.join(raw_dir, f"{ticker.replace('.', '_')}.csv")
        if not os.path.exists(fname):
            print(f"Skipping {ticker}: raw data missing")
            continue

        df = pd.read_csv(fname)
        featured = engineer_features(df)
        selected, _ = select_features(featured)

        row = selection.loc[ticker]
        algorithm = row['winner']
        model_accuracy = float(row['winner_accuracy'])

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

    return {"predictions": predictions}


def main():
    payload = assemble_payload()
    predictions = payload['predictions']

    with open('data/final_payload.json', 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Assembled payload for {len(predictions)}/50 stocks")
    print(f"Wrote data/final_payload.json")
    print(f"\nSample (first stock):")
    print(json.dumps(predictions[0], indent=2))

    up_count = sum(1 for p in predictions if p['predicted_direction'] == 'up')
    print(f"\nDirection split: {up_count} up, {len(predictions) - up_count} down")
    print(f"Confidence range: {min(p['confidence'] for p in predictions):.3f} - {max(p['confidence'] for p in predictions):.3f}")


if __name__ == "__main__":
    main()

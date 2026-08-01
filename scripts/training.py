import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def train_and_evaluate(featured_df, selected_features, test_frac=0.2, random_state=42):
    """
    Given ONE stock's engineered DataFrame and its selected feature list (from Step 2),
    performs a chronological train/test split, scales features, and trains both a
    logistic regression and a random forest classifier.

    Returns a dict with both models' test accuracy, predictions, and metadata -
    NOT a final evaluation (no baseline comparison yet - that's Step 5).
    """
    clean = featured_df.dropna(subset=selected_features + ['target']).reset_index(drop=True)

    X = clean[selected_features]
    y = clean['target'].astype(int)

    # chronological split: no shuffling, train is strictly earlier than test
    split_idx = int(len(clean) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # fit scaler on train only - test set must never influence the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
    log_reg.fit(X_train_scaled, y_train)
    lr_preds = log_reg.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)

    # --- Random Forest (regularized to avoid overfitting on ~600 training rows) ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=10,
        random_state=random_state,
    )
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)

    return {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'logistic_regression': {
            'model': log_reg,
            'test_accuracy': lr_acc,
            'predictions': lr_preds,
        },
        'random_forest': {
            'model': rf,
            'test_accuracy': rf_acc,
            'predictions': rf_preds,
        },
        'y_test': y_test.values,
        'scaler': scaler,
    }

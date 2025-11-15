#!/usr/bin/env python3
"""
Train ML model for predicting 10:15 → 10:30 continuation (NO LEAKAGE).

This version includes:
    ✔ Shuffled-label test
    ✔ Feature ↔ label correlation scan
    ✔ Duplicate row detection
    ✔ Probability sanity check
    ✔ Train/test leakage checks
    ✔ Verbose diagnostics

If any leakage is detected → raises clear RuntimeError.
"""

import os
import sys
import json
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
)
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(PROJECT_ROOT, "data", "metrics")
MODEL_OUT = os.path.join(PROJECT_ROOT, "model_weights.json")

MIN_SAMPLES = 500
BALANCE = True

# -----------------------------------------------------------
# Features (must be available BEFORE 10:15)
# -----------------------------------------------------------
FEATURES = [
    # Premarket
    "premarket_pct_change",
    "premarket_range_pct",
    "pm_trend_slope_pct_per_hour",
    "pm_pullback_depth_pct",
    "premarket_volume_60m",
    "relative_volume_20d",

    # 9:30–10:00
    "move_930_to_1000_pct",
    "price_1000_vs_opening_vwap",
    "price_1000_vs_premarket_vwap",
    "higher_highs_9_30_to_10_00",
    "higher_lows_9_30_to_10_00",
    "opening_range_high",
    "opening_range_low",

    # 10:00–10:15 micro
    "return_1000_to_1015_pct",
    "vol_spike_1000_1015_over_930_1000",
    "upper_wick_930_1015_pct",
    "lower_wick_930_1015_pct",

    # Daily
    "rsi14_daily",
    "sma20",
    "sma50",
    "atr14",

    # Market context
    "spy_ret_930_1000_pct",
    "spy_ret_1000_1015_pct",
    "qqq_ret_930_1000_pct",
    "qqq_ret_1000_1015_pct",
]


# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------

def extract_date_from_path(path: str) -> Optional[datetime.date]:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 4:
        return None
    date_str = parts[-2]
    try:
        return datetime.fromisoformat(date_str).date()
    except:
        return None


def load_all_metrics():
    print(f"[SCAN] Searching for metrics.tsv under {METRICS_DIR} ...")
    paths = glob(os.path.join(METRICS_DIR, "**", "metrics.tsv"), recursive=True)
    if not paths:
        raise RuntimeError("No metrics files found.")

    all_rows = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep="\t")
            date = extract_date_from_path(p)
            if date is None:
                continue
            df["trade_date"] = date
            df["__source_path"] = p
            all_rows.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")

    df = pd.concat(all_rows, ignore_index=True)
    print(f"[LOAD] Loaded {len(df)} rows")
    return df


# -----------------------------------------------------------
# Prepare train/test
# -----------------------------------------------------------

def prepare_training_set(df: pd.DataFrame):
    if "return_1015_to_1030_pct" not in df.columns:
        raise RuntimeError("Column 'return_1015_to_1030_pct' missing.")

    df = df.copy()
    df["label"] = (df["return_1015_to_1030_pct"] > 0).astype(int)
    df = df.dropna(subset=FEATURES + ["label", "trade_date"])

    if len(df) < MIN_SAMPLES:
        raise RuntimeError("Not enough samples for ML.")

    unique_dates = sorted(df["trade_date"].unique())
    cutoff = int(len(unique_dates) * 0.8)

    train_dates = unique_dates[:cutoff]
    test_dates = unique_dates[cutoff:]

    print(f"[SPLIT] Train {train_dates[0]} → {train_dates[-1]} ({len(train_dates)} days)")
    print(f"[SPLIT] Test  {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} days)")

    df_train = df[df["trade_date"].isin(train_dates)]
    df_test = df[df["trade_date"].isin(test_dates)]

    return (
        df_train[FEATURES].astype(float),
        df_train["label"].astype(int),
        df_test[FEATURES].astype(float),
        df_test["label"].astype(int),
        df_train,
        df_test
    )


# -----------------------------------------------------------
# Anti-leak tests
# -----------------------------------------------------------

def check_feature_correlations(X, y):
    df_corr = X.assign(label=y).corr()["label"].abs().sort_values(ascending=False)
    df_corr = df_corr.drop("label") 
    print("\n=== FEATURE ↔ LABEL CORRELATION ===")
    print(df_corr.head(20))

    if df_corr.iloc[0] > 0.5:
        raise RuntimeError(
            f"🚨 HIGH CORRELATION DETECTED: {df_corr.index[0]} correlates {df_corr.iloc[0]:.3f} with the label.\n"
            "This strongly indicates future leakage."
        )


def check_duplicates(df_train, df_test):
    dupes = pd.merge(df_train[FEATURES], df_test[FEATURES], how="inner")
    print(f"\n[DUPLICATE CHECK] duplicates between train/test: {len(dupes)}")
    if len(dupes) > 0:
        raise RuntimeError("🚨 Data leakage: train/test share identical rows!")


def shuffled_label_test(X_train, y_train):
    y_shuffled = y_train.sample(frac=1.0, random_state=42).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    pipe.fit(X_train, y_shuffled)

    acc = pipe.score(X_train, y_shuffled)
    print(f"[SHUFFLED TEST] Accuracy = {acc:.3f}")

    if acc > 0.55:
        raise RuntimeError(
            "🚨 Leakage confirmed: model can learn random labels.\n"
            "Some feature still leaks future information."
        )
    print("✅ Shuffled test passed — no leakage signal.")


def probability_sanity_check(pipe, X_test):
    p = pipe.predict_proba(X_test)[:, 1]
    if np.any((p <= 1e-6) | (p >= 1 - 1e-6)):
        raise RuntimeError(
            "🚨 Model outputted probability ≈ 0 or ≈ 1:\n"
            "This means a feature is acting as a perfect oracle.\n"
            "Leakage still exists."
        )
    print("✅ Probability range looks normal.")


# -----------------------------------------------------------
# Train + evaluate
# -----------------------------------------------------------

def train_logistic_model(X, y):
    if BALANCE:
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)
    else:
        clf = LogisticRegression(C=0.5, max_iter=2000)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipe.fit(X, y)
    return pipe


def evaluate(pipe, X_train, y_train, X_test, y_test):
    print("\n=== TRAIN PERFORMANCE ===")
    p_train = pipe.predict_proba(X_train)[:, 1]
    y_hat_train = (p_train >= 0.5)

    print("[ACC]", accuracy_score(y_train, y_hat_train))
    print("[AUC]", roc_auc_score(y_train, p_train))
    print(classification_report(y_train, y_hat_train))

    print("\n=== TEST PERFORMANCE ===")
    p_test = pipe.predict_proba(X_test)[:, 1]
    y_hat_test = (p_test >= 0.5)

    print("[ACC]", accuracy_score(y_test, y_hat_test))
    print("[AUC]", roc_auc_score(y_test, p_test))
    print(classification_report(y_test, y_hat_test))


# -----------------------------------------------------------
# Save model
# -----------------------------------------------------------

def export_model(pipe):
    scaler = pipe.named_steps["scaler"]
    clf     = pipe.named_steps["clf"]

    export = {
        "bias": float(clf.intercept_[0]),
        "weights": {feat: float(w) for feat, w in zip(FEATURES, clf.coef_[0])},
        "scaler_mean": {feat: float(m) for feat, m in zip(FEATURES, scaler.mean_)},
        "scaler_scale": {feat: float(s) for feat, s in zip(FEATURES, scaler.scale_)},
        "features": FEATURES,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    with open(MODEL_OUT, "w") as f:
        json.dump(export, f, indent=4)

    print(f"[SAVE] Model saved → {MODEL_OUT}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    df = load_all_metrics()
    X_train, y_train, X_test, y_test, df_train, df_test = prepare_training_set(df)

    # 1. Anti-leak checks
    check_feature_correlations(X_train, y_train)
    check_duplicates(df_train, df_test)
    shuffled_label_test(X_train, y_train)

    # 2. Train model
    pipe = train_logistic_model(X_train, y_train)

    # 3. Probability sanity check
    probability_sanity_check(pipe, X_test)

    # 4. Evaluate
    evaluate(pipe, X_train, y_train, X_test, y_test)

    # 5. Save weights
    export_model(pipe)

    print("[DONE] Training complete with all anti-leak checks passed.")


if __name__ == "__main__":
    main()

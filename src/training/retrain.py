"""Automated BTC-USD drift detection and retraining pipeline.

Steps
-----
1. Download full BTC-USD price history via yfinance
2. Build features (identical to app.py / notebook)
3. Evaluate the current model on the last EVAL_DAYS days  →  recent MAE
4. If recent MAE > DRIFT_THRESHOLD × baseline RMSE  →  retrain GRU
5. Save new model artifacts (best_model.keras, scaler_X.pkl, scaler_y.pkl)
6. Update models/selection.json with new RMSE
7. Write models/drift_report.json with run metadata
8. Log the run to MLflow

Usage
-----
    # Normal run (retrain only if drift detected)
    python src/training/retrain.py

    # Force retrain regardless of recent performance
    python src/training/retrain.py --force
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

# ── Paths ──────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(__file__)
MODEL_DIR      = os.path.abspath(os.path.join(_HERE, "..", "..", "models"))
SELECTION_FILE = os.path.join(MODEL_DIR, "selection.json")
REPORT_FILE    = os.path.join(MODEL_DIR, "drift_report.json")

# ── Hyper-parameters ───────────────────────────────────────────────────
SEQ_LEN         = 100   # number of lag features
LOOKBACK        = 20    # GRU sequence length
EVAL_DAYS       = 30    # recent window for drift detection
DRIFT_THRESHOLD = 1.5   # retrain if recent_MAE > threshold × baseline_RMSE


# ── Feature engineering (mirrors app.py exactly) ──────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    return 100 - (100 / (1 + up / down))


def _macd(series: pd.Series, a: int = 12, b: int = 26, c: int = 9):
    ema_a = series.ewm(span=a, adjust=False).mean()
    ema_b = series.ewm(span=b, adjust=False).mean()
    line  = ema_a - ema_b
    sig   = line.ewm(span=c, adjust=False).mean()
    return line, sig


def build_features(prices: list[float], seq_len: int = SEQ_LEN) -> pd.DataFrame:
    """Build a feature DataFrame from raw Close prices (identical to app.py)."""
    d = pd.DataFrame({"Close": prices})
    d["return"]   = d["Close"].pct_change().shift(1)
    d["std30"]    = d["Close"].rolling(30).std().shift(1)
    d["rsi14"]    = _rsi(d["Close"]).shift(1)
    ml, ms        = _macd(d["Close"])
    d["macd"]     = ml.shift(1)
    d["macd_sig"] = ms.shift(1)
    d = d.dropna().reset_index(drop=True)
    for i in range(1, seq_len + 1):
        d[f"lag_{i}"] = d["Close"].shift(i)
    d = d.dropna().reset_index(drop=True)
    return d


def _feature_cols(seq_len: int = SEQ_LEN) -> list[str]:
    lag_cols   = [f"lag_{i}" for i in range(1, seq_len + 1)]
    extra_cols = ["std30", "rsi14", "macd", "macd_sig", "return"]
    return lag_cols + extra_cols


def _gru_seq(X_scaled: np.ndarray, lookback: int) -> np.ndarray:
    """Extract GRU input sequences from scaled feature matrix.

    Columns 0…lookback-1 correspond to lag_1…lag_lookback.
    We reverse their order so the sequence runs oldest → newest.
    """
    idx = list(range(lookback - 1, -1, -1))   # [lookback-1, …, 0]
    return X_scaled[:, idx].reshape(-1, lookback, 1)


# ── Drift detection ────────────────────────────────────────────────────

def evaluate_recent(
    feat_df:  pd.DataFrame,
    model:    tf.keras.Model,
    scaler_X: MinMaxScaler,
    scaler_y: StandardScaler,
    lookback: int = LOOKBACK,
    seq_len:  int = SEQ_LEN,
    n:        int = EVAL_DAYS,
) -> tuple[float, list[float]]:
    """Predict the last *n* closing prices and return (MAE, predictions)."""
    fcols = _feature_cols(seq_len)
    eval_rows  = feat_df.iloc[-n:]
    actual     = eval_rows["Close"].values

    X     = eval_rows[fcols].values
    X_s   = scaler_X.transform(X)
    seqs  = _gru_seq(X_s, lookback)

    scaled_preds = model.predict(seqs, verbose=0).ravel()
    log_rets     = scaler_y.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
    prev_closes  = eval_rows["lag_1"].values
    predictions  = (prev_closes * np.exp(log_rets)).tolist()

    mae = float(mean_absolute_error(actual, predictions))
    return mae, predictions


# ── Retraining ─────────────────────────────────────────────────────────

def retrain(
    feat_df:  pd.DataFrame,
    lookback: int = LOOKBACK,
    seq_len:  int = SEQ_LEN,
) -> tuple[tf.keras.Model, MinMaxScaler, StandardScaler, float]:
    """Retrain GRU from scratch on *feat_df* and return (model, scaler_X, scaler_y, test_RMSE)."""
    fcols = _feature_cols(seq_len)

    X = feat_df[fcols].values
    y = np.log(feat_df["Close"].values / feat_df["lag_1"].values)  # log-return

    split     = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_X  = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)

    scaler_y  = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_train_seq = _gru_seq(X_train_s, lookback)
    X_test_seq  = _gru_seq(X_test_s,  lookback)

    model = Sequential([
        GRU(64, input_shape=(lookback, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_seq, y_train_s,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )

    # Evaluate on hold-out set
    scaled_preds = model.predict(X_test_seq, verbose=0).ravel()
    log_rets     = scaler_y.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
    prev_closes  = feat_df["lag_1"].iloc[split:].values
    actual       = feat_df["Close"].iloc[split:].values
    predicted    = prev_closes * np.exp(log_rets)
    rmse         = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    return model, scaler_X, scaler_y, rmse


# ── Main ───────────────────────────────────────────────────────────────

def main(force: bool = False) -> dict:
    # ── 1. Download data ──────────────────────────────────────────────
    print("Downloading BTC-USD history...")
    raw = yf.download("BTC-USD", start="2014-01-01", progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prices = raw["Close"].dropna().squeeze().tolist()
    print(f"  {len(prices)} daily closes  ({raw.index[0].date()} → {raw.index[-1].date()})")

    # ── 2. Build features ─────────────────────────────────────────────
    feat_df = build_features(prices)
    print(f"  {len(feat_df)} rows after feature engineering")

    # ── 3. Load current model & baseline ─────────────────────────────
    with open(SELECTION_FILE) as f:
        sel = json.load(f)

    baseline_rmse = float(sel.get("rmse", 1901.0))
    lookback      = int(sel.get("gru_lookback", LOOKBACK))
    seq_len       = len([k for k in sel.get("features", []) if k.startswith("lag_")]) or SEQ_LEN

    model    = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))

    # ── 4. Drift detection ────────────────────────────────────────────
    print(f"\nEvaluating last {EVAL_DAYS} days...")
    recent_mae, _ = evaluate_recent(feat_df, model, scaler_X, scaler_y, lookback, seq_len)
    threshold     = DRIFT_THRESHOLD * baseline_rmse
    drift         = recent_mae > threshold

    print(f"  Recent MAE:      ${recent_mae:>10,.2f}")
    print(f"  Baseline RMSE:   ${baseline_rmse:>10,.2f}")
    print(f"  Drift threshold: ${threshold:>10,.2f}  ({DRIFT_THRESHOLD}× baseline)")
    print(f"  Drift detected:  {drift}")

    report: dict = {
        "timestamp":       datetime.datetime.utcnow().isoformat(),
        "eval_days":       EVAL_DAYS,
        "recent_mae":      round(recent_mae, 2),
        "baseline_rmse":   round(baseline_rmse, 2),
        "drift_threshold": round(threshold, 2),
        "drift_detected":  drift,
        "retrained":       False,
    }

    # ── 5. Retrain if needed ──────────────────────────────────────────
    if force or drift:
        reason = "forced" if force else "drift detected"
        print(f"\nRetraining GRU ({reason})...")

        run_name = f"retrain-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M')}"

        if _MLFLOW:
            mlflow.set_experiment("bitcoin-price-prediction")
            ctx = mlflow.start_run(run_name=run_name)
        else:
            ctx = _NullContext()

        with ctx:
            if _MLFLOW:
                mlflow.log_params({
                    "trigger":  reason,
                    "lookback": lookback,
                    "seq_len":  seq_len,
                    "eval_days": EVAL_DAYS,
                })
                mlflow.log_metric("pre_retrain_mae", recent_mae)

            new_model, new_scaler_X, new_scaler_y, new_rmse = retrain(feat_df, lookback, seq_len)
            print(f"\n  New test RMSE: ${new_rmse:>10,.2f}  (was ${baseline_rmse:,.2f})")

            if _MLFLOW:
                mlflow.log_metric("new_rmse", new_rmse)

        # Save artifacts
        new_model.save(os.path.join(MODEL_DIR, "best_model.keras"))
        joblib.dump(new_scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
        joblib.dump(new_scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))

        # Update selection.json
        sel["rmse"]      = round(new_rmse, 2)
        sel["retrained"] = datetime.datetime.utcnow().isoformat()
        with open(SELECTION_FILE, "w") as f:
            json.dump(sel, f, indent=2)
        print("  Artifacts saved.")

        report["retrained"] = True
        report["new_rmse"]  = round(new_rmse, 2)

    # ── 6. Write drift report ─────────────────────────────────────────
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDrift report → {REPORT_FILE}")

    return report


class _NullContext:
    """Fallback context manager when MLflow is unavailable."""
    def __enter__(self):  return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC drift check + conditional retraining")
    parser.add_argument("--force", action="store_true", help="Retrain regardless of drift")
    args = parser.parse_args()

    result = main(force=args.force)
    print()
    print(json.dumps(result, indent=2))

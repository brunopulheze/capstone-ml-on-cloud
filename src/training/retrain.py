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
SEQ_LEN         = 20    # number of lag features (= LOOKBACK — this script trains GRU only)
LOOKBACK        = 20    # GRU sequence length
EVAL_DAYS       = 30    # recent window for drift detection
DRIFT_THRESHOLD = 1.5   # retrain if recent_MAE > threshold × baseline_RMSE


# ── Feature engineering (mirrors app.py exactly) ──────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a price series.

    RSI measures momentum by comparing the average gain to the average loss
    over a rolling window. The result is bounded between 0 and 100:
      - Values above 70 indicate the asset is overbought (price rose fast).
      - Values below 30 indicate the asset is oversold (price fell fast).

    Parameters
    ----------
    series : pd.Series
        Raw closing prices.
    period : int
        Look-back window in days. Default is 14 (Wilder's original setting).

    Returns
    -------
    pd.Series
        RSI values aligned to the same index as *series*.
    """
    delta = series.diff()                           # daily change: close[t] - close[t-1]
    up    = delta.clip(lower=0).rolling(period).mean()   # average gain over the window
    down  = -delta.clip(upper=0).rolling(period).mean()  # average loss (made positive)
    return 100 - (100 / (1 + up / down))           # standard RSI formula


def _macd(series: pd.Series, a: int = 12, b: int = 26, c: int = 9):
    """Compute the MACD line and signal line for a price series.

    MACD (Moving Average Convergence/Divergence) is a trend-following indicator
    that measures the gap between a fast and a slow exponential moving average.
      - MACD line > 0: short-term momentum is above long-term (bullish).
      - MACD line < 0: short-term momentum is below long-term (bearish).
      - A crossover of the MACD line above/below the signal line is a classic
        buy/sell signal used in technical analysis.

    Parameters
    ----------
    series : pd.Series
        Raw closing prices.
    a : int
        Span for the fast EMA. Default 12 days.
    b : int
        Span for the slow EMA. Default 26 days.
    c : int
        Span for the signal line EMA. Default 9 days.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (macd_line, signal_line)
    """
    ema_a = series.ewm(span=a, adjust=False).mean()  # fast EMA (12-day)
    ema_b = series.ewm(span=b, adjust=False).mean()  # slow EMA (26-day)
    line  = ema_a - ema_b                             # MACD line: fast minus slow
    sig   = line.ewm(span=c, adjust=False).mean()    # signal line: 9-day EMA of MACD
    return line, sig


def build_features(prices: list[float], seq_len: int = SEQ_LEN) -> pd.DataFrame:
    """Build a feature DataFrame from raw Close prices (identical to app.py).

    Constructs the full 105-column feature matrix used for training and inference.
    Every indicator is shifted by 1 day (.shift(1)) so that when predicting day t,
    only information known up to day t-1 is used — preventing lookahead leakage.

    Features produced:
      - return    : yesterday's percentage change in Close
      - std30     : 30-day rolling standard deviation (volatility proxy)
      - rsi14     : RSI over 14 days (momentum oscillator)
      - macd      : MACD line — EMA(12) minus EMA(26)
      - macd_sig  : signal line — 9-day EMA of the MACD line
      - lag_1…lag_N : closing prices for the previous 1 to seq_len days

    Parameters
    ----------
    prices : list[float]
        Raw daily BTC-USD closing prices, oldest first.
    seq_len : int
        Number of lag features to generate. Default is SEQ_LEN (20).

    Returns
    -------
    pd.DataFrame
        Feature matrix with all NaN rows dropped. Each row represents one
        trading day and contains all features needed to predict the next day.
    """
    d = pd.DataFrame({"Close": prices})
    d["return"]   = d["Close"].pct_change().shift(1)   # previous day's % change
    d["std30"]    = d["Close"].rolling(30).std().shift(1)  # 30-day volatility
    d["rsi14"]    = _rsi(d["Close"]).shift(1)          # momentum oscillator
    ml, ms        = _macd(d["Close"])
    d["macd"]     = ml.shift(1)                        # trend direction
    d["macd_sig"] = ms.shift(1)                        # smoothed MACD
    d = d.dropna().reset_index(drop=True)              # drop NaN rows from indicators
    for i in range(1, seq_len + 1):
        d[f"lag_{i}"] = d["Close"].shift(i)            # lag_1=yesterday, lag_2=2 days ago…
    d = d.dropna().reset_index(drop=True)              # drop NaN rows from lag window
    return d


def _feature_cols(seq_len: int = SEQ_LEN) -> list[str]:
    """Return the ordered list of feature column names used by the model.

    The order matters: lag columns come first (indices 0…seq_len-1), which is
    assumed by _gru_seq() when it slices out the GRU lookback window.

    Parameters
    ----------
    seq_len : int
        Number of lag features. Default is SEQ_LEN (20).

    Returns
    -------
    list[str]
        Column names in the order the model expects them.
    """
    lag_cols   = [f"lag_{i}" for i in range(1, seq_len + 1)]  # lag_1, lag_2, …, lag_20
    extra_cols = ["std30", "rsi14", "macd", "macd_sig", "return"]  # technical indicators
    return lag_cols + extra_cols  # lags first — required by _gru_seq()


def _gru_seq(X_scaled: np.ndarray, lookback: int) -> np.ndarray:
    """Reshape the scaled feature matrix into 3-D sequences for the GRU layer.

    The GRU expects input of shape (samples, timesteps, features).
    Columns 0…lookback-1 in X_scaled correspond to lag_1…lag_lookback
    (lag_1 = most recent, lag_lookback = oldest). We reverse their order
    so the sequence is presented chronologically: oldest timestep first,
    most recent timestep last — the natural reading direction for a RNN.

    Example with lookback=3:
      columns in X_scaled : [lag_1, lag_2, lag_3, std30, rsi14, …]
      GRU sequence output : [[lag_3], [lag_2], [lag_1]]  (oldest → newest)

    Parameters
    ----------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix produced by MinMaxScaler.
    lookback : int
        Number of timesteps to feed into the GRU (LOOKBACK = 20).

    Returns
    -------
    np.ndarray, shape (n_samples, lookback, 1)
        3-D array ready to pass to model.predict() or model.fit().
    """
    idx = list(range(lookback - 1, -1, -1))   # reverse: [19, 18, …, 0] for lookback=20
    return X_scaled[:, idx].reshape(-1, lookback, 1)  # → (n, 20, 1)


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
    """Run the current model on the most recent *n* days and measure its error.

    This is the drift detection step. We predict each of the last n closing
    prices using the existing model, compare to the actual closes, and return
    the Mean Absolute Error (MAE) in USD. If this MAE exceeds
    DRIFT_THRESHOLD × baseline_RMSE, the model is considered to have drifted
    and a retrain is triggered.

    The full prediction pipeline mirrors app.py:
      feature row → MinMaxScaler → GRU sequence → model.predict()
      → StandardScaler inverse → exp() back-transform → USD price

    Parameters
    ----------
    feat_df : pd.DataFrame
        Full feature matrix produced by build_features().
    model : tf.keras.Model
        The currently deployed GRU model.
    scaler_X : MinMaxScaler
        Fitted feature scaler (loaded from scaler_X.pkl).
    scaler_y : StandardScaler
        Fitted target scaler (loaded from scaler_y.pkl).
    lookback : int
        GRU sequence length. Default LOOKBACK (20).
    seq_len : int
        Number of lag features. Default SEQ_LEN (20).
    n : int
        Number of recent days to evaluate. Default EVAL_DAYS (30).

    Returns
    -------
    tuple[float, list[float]]
        (MAE in USD, list of predicted prices for the n days)
    """
    fcols = _feature_cols(seq_len)
    eval_rows  = feat_df.iloc[-n:]          # last n rows of the feature matrix
    actual     = eval_rows["Close"].values  # true closing prices to compare against

    X     = eval_rows[fcols].values
    X_s   = scaler_X.transform(X)           # scale features to [0, 1]
    seqs  = _gru_seq(X_s, lookback)         # reshape to (n, 20, 1) for GRU

    scaled_preds = model.predict(seqs, verbose=0).ravel()  # raw model output (scaled)
    log_rets     = scaler_y.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()  # un-standardise
    prev_closes  = eval_rows["lag_1"].values
    predictions  = (prev_closes * np.exp(log_rets)).tolist()  # back-transform to USD price

    mae = float(mean_absolute_error(actual, predictions))  # average absolute error in USD
    return mae, predictions


# ── Retraining ─────────────────────────────────────────────────────────

def retrain(
    feat_df:  pd.DataFrame,
    lookback: int = LOOKBACK,
    seq_len:  int = SEQ_LEN,
) -> tuple[tf.keras.Model, MinMaxScaler, StandardScaler, float]:
    """Train a new GRU model from scratch on the full price history.

    Called when drift is detected or --force is passed. Fits fresh scalers
    on the training split (so they reflect the current price range), trains
    the GRU with early stopping to avoid overfitting, then evaluates RMSE
    on the held-out 30% test set.

    Training details:
      - Target   : log-return  log(close[t] / close[t-1])
      - Split    : chronological 70 / 30 (no shuffling — avoids leakage)
      - Scaling  : MinMaxScaler on X, StandardScaler on y
      - Model    : GRU(64) → Dense(1), Adam optimiser, MSE loss
      - Stopping : EarlyStopping with patience=10, restores best weights

    Parameters
    ----------
    feat_df : pd.DataFrame
        Full feature matrix produced by build_features().
    lookback : int
        GRU sequence length (number of timesteps). Default LOOKBACK (20).
    seq_len : int
        Number of lag features. Default SEQ_LEN (20).

    Returns
    -------
    tuple[tf.keras.Model, MinMaxScaler, StandardScaler, float]
        (trained model, feature scaler, target scaler, test RMSE in USD)
    """
    fcols = _feature_cols(seq_len)

    X = feat_df[fcols].values
    y = np.log(feat_df["Close"].values / feat_df["lag_1"].values)  # log-return target

    # Chronological 70/30 split — never shuffle time-series data
    split     = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit scalers on training data only — never on test data (would leak future info)
    scaler_X  = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)  # fit+transform on train
    X_test_s  = scaler_X.transform(X_test)        # transform only on test

    scaler_y  = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # standardise log-returns

    # Reshape to 3-D sequences required by the GRU layer: (samples, timesteps, 1)
    X_train_seq = _gru_seq(X_train_s, lookback)
    X_test_seq  = _gru_seq(X_test_s,  lookback)

    # Build the GRU architecture
    model = Sequential([
        GRU(64, input_shape=(lookback, 1)),  # 64 hidden units, reads 20-step sequences
        Dense(1),                            # single output: predicted log-return
    ])
    model.compile(optimizer="adam", loss="mse")  # Adam + MSE is standard for regression
    model.fit(
        X_train_seq, y_train_s,
        epochs=100,           # upper bound — early stopping will usually kick in earlier
        batch_size=32,
        validation_split=0.1, # 10% of training data used to monitor val loss
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )

    # Evaluate on the held-out test set
    scaled_preds = model.predict(X_test_seq, verbose=0).ravel()
    log_rets     = scaler_y.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()  # un-standardise
    prev_closes  = feat_df["lag_1"].iloc[split:].values
    actual       = feat_df["Close"].iloc[split:].values
    predicted    = prev_closes * np.exp(log_rets)  # price[t] = close[t-1] × exp(log_return)

    # Price RMSE — kept only for retrain drift detection (not a skill metric)
    rmse         = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    # Log-return RMSE — the honest skill metric
    y_test_logret = y[split:]
    logret_rmse   = float(np.sqrt(np.mean((y_test_logret - log_rets) ** 2)))

    return model, scaler_X, scaler_y, rmse, logret_rmse


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

    # ── 3. Load baseline config ───────────────────────────────────────
    sel: dict = {}
    if os.path.exists(SELECTION_FILE):
        with open(SELECTION_FILE) as f:
            sel = json.load(f)

    baseline_rmse = float(sel.get("rmse", 1901.0))
    lookback      = int(sel.get("gru_lookback", LOOKBACK))
    seq_len       = len([k for k in sel.get("features", []) if k.startswith("lag_")]) or SEQ_LEN

    # ── 4. Drift detection (skipped on cold start) ────────────────────
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    cold_start = not os.path.exists(model_path)

    if cold_start:
        print("\nNo existing model found — cold-start training.")
        recent_mae: float | None = None
        drift = False
        force = True   # always train when no model exists
    else:
        model    = tf.keras.models.load_model(model_path)
        scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
        scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))

        print(f"\nEvaluating last {EVAL_DAYS} days...")
        recent_mae, _ = evaluate_recent(feat_df, model, scaler_X, scaler_y, lookback, seq_len)
        threshold     = DRIFT_THRESHOLD * baseline_rmse
        drift         = recent_mae > threshold

        print(f"  Recent MAE:      ${recent_mae:>10,.2f}")
        print(f"  Baseline RMSE:   ${baseline_rmse:>10,.2f}")
        print(f"  Drift threshold: ${threshold:>10,.2f}  ({DRIFT_THRESHOLD}× baseline)")
        print(f"  Drift detected:  {drift}")

    threshold = DRIFT_THRESHOLD * baseline_rmse
    report: dict = {
        "timestamp":       datetime.datetime.utcnow().isoformat(),
        "eval_days":       EVAL_DAYS,
        "recent_mae":      round(recent_mae, 2) if recent_mae is not None else None,
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
                if recent_mae is not None:
                    mlflow.log_metric("pre_retrain_mae", recent_mae)

            new_model, new_scaler_X, new_scaler_y, new_rmse, new_logret_rmse = retrain(feat_df, lookback, seq_len)
            print(f"\n  New price RMSE:   ${new_rmse:>10,.2f}  (was ${baseline_rmse:,.2f})")
            print(f"  New logret RMSE:  {new_logret_rmse:.5f}")

            if _MLFLOW:
                mlflow.log_metric("new_rmse",       new_rmse)
                mlflow.log_metric("logret_rmse",    new_logret_rmse)

        # Save artifacts
        new_model.save(os.path.join(MODEL_DIR, "best_model.keras"))
        joblib.dump(new_scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
        joblib.dump(new_scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))

        # Update selection.json
        sel["rmse"]         = round(new_rmse, 2)
        sel["logret_rmse"]  = round(new_logret_rmse, 5)
        sel["retrained"]    = datetime.datetime.utcnow().isoformat()
        with open(SELECTION_FILE, "w") as f:
            json.dump(sel, f, indent=2)
        print("  Artifacts saved.")

        report["retrained"]    = True
        report["new_rmse"]     = round(new_rmse, 2)
        report["new_logret_rmse"] = round(new_logret_rmse, 5)

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

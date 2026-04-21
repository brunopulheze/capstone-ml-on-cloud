"""
BTC Price Predictor — FastAPI app (GRU model)

POST /predict  {"prices": [<list of daily Close prices, oldest first>]}
  → {"predicted_price": <float>, "previous_close": <float>}

The `prices` list must contain at least SEQ_LEN + 200 values so that all
technical indicators (RSI-14, MACD-26, rolling-std-30, 100-day lags) can
be computed without NaN rows after the warmup period.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BTC Price Predictor — GRU")

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR      = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
SELECTION_FILE = os.path.join(MODEL_DIR, "selection.json")

# ── Global state (populated at startup) ──────────────────────────────
_gru_model  = None
_scaler_X   = None
_scaler_y   = None
_lookback   = 20
_seq_len    = 100


# ── Feature engineering (mirrors the notebook) ───────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    return 100 - (100 / (1 + up / down))


def _macd(series: pd.Series, a=12, b=26, c=9):
    ema_a = series.ewm(span=a, adjust=False).mean()
    ema_b = series.ewm(span=b, adjust=False).mean()
    line  = ema_a - ema_b
    sig   = line.ewm(span=c, adjust=False).mean()
    return line, sig


def _build_features(prices: list[float], seq_len: int) -> pd.DataFrame:
    """Build a single-row feature DataFrame from a list of raw Close prices."""
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


# ── Startup ───────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global _gru_model, _scaler_X, _scaler_y, _lookback, _seq_len

    if not os.path.exists(SELECTION_FILE):
        raise RuntimeError(f"selection.json not found at {SELECTION_FILE}. Run the notebook first.")

    with open(SELECTION_FILE) as f:
        sel = json.load(f)

    _lookback = sel.get("gru_lookback", 20)
    _seq_len  = len([k for k in sel.get("features", []) if k.startswith("lag_")])
    if _seq_len == 0:
        _seq_len = 100

    import tensorflow as tf
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    _gru_model = tf.keras.models.load_model(model_path)

    _scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    _scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": "GRU", "lookback": _lookback}


@app.get("/health")
def health():
    return {"status": "healthy"}


class PredictRequest(BaseModel):
    prices: list[float]  # daily Close prices, oldest first; min SEQ_LEN+200 values


@app.post("/predict")
def predict(payload: PredictRequest):
    min_len = _seq_len + 200  # warmup for RSI/MACD/std
    if len(payload.prices) < min_len:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_len} prices for indicator warmup (got {len(payload.prices)})"
        )

    feat_df = _build_features(payload.prices, seq_len=_seq_len)
    if feat_df.empty:
        raise HTTPException(status_code=422, detail="Not enough data after feature engineering — send more prices.")

    lag_cols     = [f"lag_{i}" for i in range(1, _seq_len + 1)]
    extra_cols   = ["std30", "rsi14", "macd", "macd_sig", "return"]
    feature_cols = lag_cols + extra_cols

    # Use the last available row (most recent prediction point)
    row = feat_df[feature_cols].iloc[[-1]]
    row_s = _scaler_X.transform(row)

    # Build GRU sequence: LOOKBACK timesteps of lag prices (oldest → newest)
    gru_lag_idx = list(range(_lookback - 1, -1, -1))
    seq = row_s[:, gru_lag_idx].reshape(1, _lookback, 1)

    scaled_pred = _gru_model.predict(seq, verbose=0).ravel()
    log_ret     = _scaler_y.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()[0]
    prev_close  = float(feat_df["lag_1"].iloc[-1])
    predicted   = float(prev_close * np.exp(log_ret))

    return {
        "predicted_price": round(predicted, 2),
        "previous_close":  round(prev_close, 2),
    }

    # Expect either flat input for LR/RF (length=SEQ_LEN) or last sequence for LSTM
    if MODEL_TYPE_LOADED in ('lstm', 'keras'):
        # reshape to (1, timesteps, 1)
        x = series.reshape(1, -1, 1)
        pred = MODEL_OBJ.predict(x)
        if SCALER is not None:
            pred = SCALER.inverse_transform(pred.reshape(-1, 1)).flatten()[0]
        else:
            pred = float(pred.flatten()[0])
    else:
        x = series.reshape(1, -1)
        pred = MODEL_OBJ.predict(x)
        if SCALER is not None:
            pred = SCALER.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
        else:
            pred = float(np.array(pred).flatten()[0])

    return {'prediction': float(pred), 'model': MODEL_TYPE_LOADED}

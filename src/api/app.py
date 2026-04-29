"""
BTC Price Predictor — FastAPI app (Random Forest model)

GET  /predict/latest
    → fetches live BTC-USD prices via yfinance, returns tomorrow's prediction
        {"predicted_price": <float>, "previous_close": <float>,
        "last_data_date": <str>, "data_points": <int>}

POST /predict  {"prices": [<list of daily Close prices, oldest first>]}
    → {"predicted_price": <float>, "previous_close": <float>}

The `prices` list must contain at least SEQ_LEN + 200 values so that all
technical indicators (RSI-14, MACD-26, rolling-std-30, 20-day lags) can
be computed without NaN rows after the warmup period.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="BTC Price Predictor — Random Forest")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR      = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
SELECTION_FILE = os.path.join(MODEL_DIR, "selection.json")

# ── Global state (populated at startup) ──────────────────────────────
_rf_model   = None
_scaler_X   = None
_scaler_y   = None
_seq_len    = 20


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
    global _rf_model, _scaler_X, _scaler_y, _seq_len

    if not os.path.exists(SELECTION_FILE):
        raise RuntimeError(f"selection.json not found at {SELECTION_FILE}. Run the notebook first.")

    with open(SELECTION_FILE) as f:
        sel = json.load(f)

    _seq_len = len([k for k in sel.get("features", []) if k.startswith("lag_")])
    if _seq_len == 0:
        _seq_len = 20

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    _rf_model = joblib.load(model_path)

    _scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    _scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": "RF", "n_features": _seq_len + 5}


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

    scaled_pred = _rf_model.predict(row_s).ravel()
    log_ret     = _scaler_y.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()[0]
    prev_close  = float(feat_df["lag_1"].iloc[-1])
    predicted   = float(prev_close * np.exp(log_ret))

    return {
        "predicted_price": round(predicted, 2),
        "previous_close":  round(prev_close, 2),
    }


@app.get("/predict/latest")
def predict_latest():
    """Autonomous endpoint — fetches live BTC-USD prices via yfinance and returns tomorrow's prediction."""
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(status_code=500, detail="yfinance not installed in this environment.")

    df = yf.download("BTC-USD", start="2014-01-01", progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=503, detail="Failed to fetch BTC-USD data from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prices = df["Close"].dropna().squeeze().tolist()

    min_len = _seq_len + 200
    if len(prices) < min_len:
        raise HTTPException(status_code=503, detail=f"Not enough historical data (got {len(prices)}, need {min_len}).")

    feat_df = _build_features(prices, seq_len=_seq_len)
    if feat_df.empty:
        raise HTTPException(status_code=422, detail="Not enough data after feature engineering.")

    lag_cols     = [f"lag_{i}" for i in range(1, _seq_len + 1)]
    extra_cols   = ["std30", "rsi14", "macd", "macd_sig", "return"]
    feature_cols = lag_cols + extra_cols

    row   = feat_df[feature_cols].iloc[[-1]]
    row_s = _scaler_X.transform(row)

    scaled_pred = _rf_model.predict(row_s).ravel()
    log_ret     = _scaler_y.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()[0]
    prev_close  = float(feat_df["lag_1"].iloc[-1])
    predicted   = float(prev_close * np.exp(log_ret))

    last_date = df.index[-1].strftime("%Y-%m-%d")
    return {
        "predicted_price": round(predicted, 2),
        "previous_close":  round(prev_close, 2),
        "last_data_date":  last_date,
        "data_points":     len(prices),
    }


@app.get("/drift-report")
def drift_report():
    """Return the latest drift detection report, or a default if not yet generated."""
    report_path = os.path.join(MODEL_DIR, "drift_report.json")
    if not os.path.exists(report_path):
        return {
            "available": False,
            "message": "No drift report yet — awaiting first scheduled GitHub Actions run (daily at 06:00 UTC).",
        }
    with open(report_path) as f:
        data = json.load(f)
    data["available"] = True
    return data

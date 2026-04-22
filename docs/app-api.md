# BTC Price Predictor — API Reference (`src/api/app.py`)

## Overview

`app.py` is a **FastAPI** application that serves the trained GRU model as an HTTP REST API. It accepts a list of historical Bitcoin close prices, runs the same feature engineering pipeline used during training, and returns a next-day price prediction.

**Live URL**: http://138.2.180.250:8080

```
GET  /health          → {"status": "healthy"}
GET  /                → {"status": "ok", "model": "GRU", "lookback": 20}
GET  /predict/latest  → autonomous prediction (no input required)
POST /predict         → prediction from user-supplied price list
```

---

## Startup sequence

When the server starts (`uvicorn src.api.app:app`), the `startup()` function runs automatically and:

1. Reads `models/selection.json` to get `gru_lookback` and the feature list
2. Loads `models/best_model.keras` — the trained GRU network (TensorFlow/Keras)
3. Loads `models/scaler_X.pkl` — `MinMaxScaler` fitted on training features
4. Loads `models/scaler_y.pkl` — `StandardScaler` fitted on training log-returns

The model directory defaults to `../../models` relative to `app.py`, and can be overridden with the `MODEL_DIR` environment variable (used by the Docker container).

---

## Endpoints

### `GET /`
Returns the server status and the GRU lookback window size.

**Response**
```json
{"status": "ok", "model": "GRU", "lookback": 20}
```

---

### `GET /health`
Lightweight liveness check for load balancers and monitoring.

**Response**
```json
{"status": "healthy"}
```

---

### `GET /predict/latest`
Autonomous endpoint — no input required. Downloads the full BTC-USD price history via **yfinance**, runs the prediction pipeline internally, and returns tomorrow's estimated price.

**Response**
```json
{
  "predicted_price": 75684.95,
  "previous_close":  76352.77,
  "last_data_date":  "2026-04-22",
  "data_points":     4236
}
```

| Field | Description |
|-------|-------------|
| `predicted_price` | GRU's estimate of the next-day close price (USD) |
| `previous_close` | Most recent close price used as the reconstruction base |
| `last_data_date` | Date of the most recent candle used |
| `data_points` | Total number of price records downloaded |

---

### `POST /predict`
Runs the full prediction pipeline and returns tomorrow's estimated BTC price.

**Request body**
```json
{
  "prices": [<list of daily Close prices, oldest first>]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prices` | `list[float]` | Historical daily BTC/USD close prices, **oldest first**. Must contain at least `SEQ_LEN + 200 = 300` values to allow all indicators to warm up without NaN rows. In practice, send all available history (e.g. 2014–today via yfinance). |

**Response**
```json
{
  "predicted_price": 75212.93,
  "previous_close":  75872.52
}
```

| Field | Description |
|-------|-------------|
| `predicted_price` | GRU's estimate of the next-day close price (USD) |
| `previous_close` | The most recent close price used as the reconstruction base |

**Error responses**

| Code | Reason |
|------|--------|
| `422` | Fewer than 300 prices provided, or not enough data after feature engineering |

---

## Prediction pipeline (step by step)

```
prices (raw list)
      │
      ▼
_build_features()
  ├─ yesterday's return       pct_change().shift(1)
  ├─ rolling std (30-day)     rolling(30).std().shift(1)
  ├─ RSI-14                   _rsi().shift(1)
  ├─ MACD line & signal       _macd().shift(1)
  └─ lag_1 … lag_100          Close.shift(1..100)
      │
      ▼  (last row only — most recent prediction point)
scaler_X.transform()          MinMaxScaler → [0, 1]
      │
      ▼
GRU sequence                  columns lag_19…lag_0 reshaped to (1, 20, 1)
      │                       (oldest timestep first)
      ▼
gru_model.predict()           → scaled log-return
      │
      ▼
scaler_y.inverse_transform()  → raw log-return
      │
      ▼
price = prev_close × exp(log_return)
```

All indicators are **shifted by 1 day** so that at prediction time for day `t`, only information up to day `t-1` is used — no lookahead leakage.

---

## Running locally

```powershell
# From project root, with venv active
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
```

Test with real data:
```python
import yfinance as yf, requests

df = yf.download('BTC-USD', start='2014-01-01', progress=False)
df.columns = df.columns.get_level_values(0)   # flatten MultiIndex
prices = df['Close'].dropna().tolist()

r = requests.post('http://localhost:8080/predict', json={'prices': prices})
print(r.json())
# → {'predicted_price': 75212.93, 'previous_close': 75872.52}
```

## Running in Docker

```powershell
docker build -t btc-predictor:latest .
docker run --rm -p 8081:8080 -v "${PWD}/models:/app/models" btc-predictor:latest
```

The `Dockerfile` sets `MODEL_DIR=/app/models` and starts uvicorn on port 8080.

## Live deployment (Oracle Cloud)

The API runs in a Docker container on an OCI `VM.Standard.E2.1.Micro` instance in Frankfurt.  
See [`deploy-oracle.md`](deploy-oracle.md) for the full deployment guide.

Smoke test against the live service:

```powershell
python tests/smoke_test.py --url http://138.2.180.250:8080
```

---

## Required model artifacts

All produced by running `bitcoin-price-prediction.ipynb` end-to-end:

| File | Description |
|------|-------------|
| `models/best_model.keras` | Trained GRU network |
| `models/scaler_X.pkl` | MinMaxScaler for features |
| `models/scaler_y.pkl` | StandardScaler for log-return target |
| `models/selection.json` | Metadata: lookback, feature list, RMSE |

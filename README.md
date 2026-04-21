# Capstone 2 — Bitcoin Price Prediction (ML on Cloud)

## Overview

End-to-end machine learning project that predicts daily **Bitcoin/USD (BTC-USD)** closing prices using a **GRU (Gated Recurrent Unit)** neural network. Experiment tracking is handled by MLflow. The model is served as a REST API, containerised with Docker, and deployed on **Render**.

**Live API**: https://btc-predictor-ngio.onrender.com

```
GET  /health   → {"status": "healthy"}
GET  /         → {"status": "ok", "model": "GRU", "lookback": 20}
POST /predict  → {"predicted_price": 75212.93, "previous_close": 75872.52}
```

> The free-tier Render service spins down after 15 min of inactivity — the first request after a cold start may take ~30 s.

---

## Project Structure

```
capstone-ml-on-cloud/
├── notebooks/
│   ├── bitcoin-price-prediction.ipynb     # Main notebook (EDA → GRU → evaluation)
│   └── compare_models.ipynb               # Model comparison (LR / RF / LSTM / GRU)
├── models/
│   ├── best_model.keras                   # Trained GRU model
│   ├── scaler_X.pkl                       # MinMaxScaler for features
│   ├── scaler_y.pkl                       # StandardScaler for log-return target
│   └── selection.json                     # Model metadata (type, lookback, features)
├── src/
│   ├── data/                              # Data pipeline scripts
│   ├── training/                          # Model training scripts
│   └── api/
│       └── app.py                         # FastAPI inference service
├── tests/
│   └── smoke_test.py                      # API smoke test (local or remote)
├── docs/
│   ├── capstone-briefing.md               # Project briefing
│   ├── app-api.md                         # API reference
│   └── deploy-render.md                   # Render deployment guide
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/brunopulheze/capstone-ml-on-cloud.git
cd capstone-ml-on-cloud
```

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Run the notebook
Open `notebooks/bitcoin-price-prediction.ipynb` in VS Code or Jupyter and run all cells. This trains the GRU and saves model artifacts to `models/`.

### 5. View MLflow experiments
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/mlruns
```
Then open http://localhost:5000 in your browser.

---

## ML Pipeline

| Step | Description |
|------|-------------|
| Data retrieval | Daily BTC/USD prices via `yfinance` from 2014 to today |
| Feature engineering | 100-day lag window, RSI-14, MACD, 30-day rolling std, yesterday's return — all shifted by 1 day (leak-free) |
| Preprocessing | MinMaxScaler on features, StandardScaler on log-return target, 70/30 train/test split |
| Model | **GRU(64) + Dense(1)**, LOOKBACK=20 timesteps, Adam + MSE, EarlyStopping (patience=10) |
| Tracking | MLflow logs parameters, RMSE, and artifacts |
| Evaluation | Price reconstruction from predicted log-returns, RMSE ~$1901 |

---

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | GRU(64) → Dense(1) |
| Lookback window | 20 days |
| Feature set | 100 lag features + RSI-14 + MACD + MACD signal + rolling std(30) + yesterday's return |
| Target | Log-return (rescaled back to price) |
| Test RMSE | ~$1,901 |

---

## Running the API locally

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
```

### Running in Docker

```powershell
docker build -t btc-predictor:latest .
docker run --rm -p 8081:8080 -v "${PWD}/models:/app/models" btc-predictor:latest
```

### Smoke test

```powershell
# Against local Docker container (port 8081)
python tests/smoke_test.py --port 8081

# Against the live Render deployment
python tests/smoke_test.py --url https://btc-predictor-ngio.onrender.com
```

---

## Deployment

The API is deployed on **Render** using the Docker image `brunopulheze/btc-predictor:latest` (Docker Hub).  
See [`docs/deploy-render.md`](docs/deploy-render.md) for the full deployment guide.

---

## Requirements

See `requirements.txt`. Key libraries:

- `tensorflow==2.21.0` + `keras==3.14.0` — GRU model (requires Python ≥ 3.11)
- `scikit-learn` — preprocessing and metrics
- `yfinance` — Bitcoin price data
- `mlflow` — experiment tracking
- `fastapi` + `uvicorn` — inference API
- `joblib` — scaler serialisation

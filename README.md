# Capstone 2 — Bitcoin Price Prediction (ML on Cloud)

## Overview

End-to-end machine learning project that predicts daily **Bitcoin/USD (BTC-USD)** closing prices using a **GRU (Gated Recurrent Unit)** neural network. Experiment tracking is handled by MLflow. The model is served as a REST API, containerised with Docker, and deployed on **Oracle Cloud Infrastructure**.

**Live API**: http://138.2.180.250:8080

```
GET  /health          → {"status": "healthy"}
GET  /                → {"status": "ok", "model": "GRU", "lookback": 20}
GET  /predict/latest  → autonomous prediction (fetches BTC prices via yfinance)
POST /predict         → prediction from a user-supplied price list
```

---

## Project Structure

```
capstone-ml-on-cloud/
├── notebooks/
│   ├── 01-compare-models.ipynb            # Model comparison (LR / RF / LSTM / GRU)
│   ├── 02-bitcoin-price-prediction.ipynb  # Main notebook (EDA → GRU → evaluation)
│   └── 03-feature-exploration.ipynb       # Feature exploration and analysis
├── models/
│   ├── best_model.keras                   # Trained GRU model (Keras)
│   ├── best_model.pkl                     # Best model (pickle)
│   ├── keras_model.h5                     # GRU model (HDF5 format)
│   ├── lr_model.save                      # Logistic Regression model
│   ├── rf_model.save                      # Random Forest model
│   ├── scaler.save                        # Scaler (joblib)
│   ├── scaler_X.pkl                       # MinMaxScaler for features
│   ├── scaler_y.pkl                       # StandardScaler for log-return target
│   └── selection.json                     # Model metadata (type, lookback, features)
├── src/
│   ├── data/                              # Data pipeline scripts
│   ├── training/                          # Model training scripts
│   └── api/
│       └── app.py                         # FastAPI inference service
├── dashboard/                             # Next.js 15 frontend dashboard
│   ├── app/                               # App Router pages and layout
│   └── components/
│       └── Dashboard.tsx                  # Main dashboard component
├── tests/
│   └── smoke_test.py                      # API smoke test (local or remote)
├── docs/
│   ├── capstone-briefing.md               # Project briefing
│   ├── app-api.md                         # API reference
│   ├── dashboard.md                       # Dashboard guide
│   ├── deploy-oracle.md                   # Oracle Cloud deployment guide
│   ├── model-overview.md                  # Model architecture overview
│   ├── retrain.md                         # Retraining guide
│   └── scripts-guide.md                   # Scripts reference
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

### 4. Run the notebooks
Open `notebooks/02-bitcoin-price-prediction.ipynb` in VS Code or Jupyter and run all cells. This trains the GRU and saves model artifacts to `models/`. Use `notebooks/01-compare-models.ipynb` to compare LR, RF, LSTM, and GRU models.

### 5. View MLflow experiments
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/mlruns
```
Then open http://localhost:5000 in your browser.

---

## ML Pipeline

| Step | Description |
|------|-------------|
| Data retrieval | Daily BTC/USD prices via `yfinance` from 2015 to today |
| Feature engineering | 20-day lag window, RSI-14, MACD, 30-day rolling std, yesterday's return — all shifted by 1 day (leak-free) |
| Preprocessing | MinMaxScaler on features, StandardScaler on log-return target, 70/30 train/test split |
| Model | **GRU(64) + Dense(1)**, LOOKBACK=20 timesteps, Adam + MSE, EarlyStopping (patience=10) |
| Tracking | MLflow logs parameters, RMSE, and artifacts |
| Evaluation | Price space RMSE + log-return space R² & directional accuracy, benchmarked against a persistence baseline — see [docs/honest-evaluation.md](docs/honest-evaluation.md) |

---

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | GRU(64) → Dense(1) |
| Lookback window | 20 days |
| Feature set | 20 lag features + RSI-14 + MACD + MACD signal + rolling std(30) + yesterday's return |
| Target | Log-return (rescaled back to price) |
| Test RMSE | ~$622 (GRU) — near-equal to persistence baseline; see [docs/honest-evaluation.md](docs/honest-evaluation.md) |
| Log-return R² | ≈ 0 (expected — consistent with EMH for daily BTC with price-only features) |

---

## Dashboard

A live Next.js 15 dashboard visualises actual vs predicted BTC prices using Recharts and Tailwind CSS.

```bash
cd dashboard
npm install
npm run dev
```

Then open http://localhost:3000. See [`docs/dashboard.md`](docs/dashboard.md) for details.

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

# Against the live Oracle Cloud deployment
python tests/smoke_test.py --url http://138.2.180.250:8080
```

---

## Deployment

The Docker image `brunopulheze/btc-predictor:latest` is hosted on Docker Hub and deployed on **Oracle Cloud Infrastructure** — an Always Free `VM.Standard.E2.1.Micro` instance in Frankfurt (`eu-frankfurt-1`).

| Guide | Platform | URL |
|-------|----------|-----|
| [`docs/deploy-oracle.md`](docs/deploy-oracle.md) | Oracle Cloud VM | http://138.2.180.250:8080 |

---

## Requirements

See `requirements.txt`. Key libraries:

- `tensorflow==2.21.0` + `keras==3.14.0` — GRU model (requires Python ≥ 3.11)
- `scikit-learn` — preprocessing and metrics
- `yfinance` — Bitcoin price data
- `mlflow` — experiment tracking
- `fastapi` + `uvicorn` — inference API
- `joblib` — scaler serialisation

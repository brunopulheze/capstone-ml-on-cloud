# Capstone 2 — Bitcoin Price Prediction (ML on Cloud)

## Overview

End-to-end machine learning project that predicts daily **Bitcoin/USD (BTC-USD)** closing prices using a **Random Forest** model trained on log-returns. Experiment tracking is handled by MLflow. The model is served as a REST API, containerised with Docker, and deployed on **Oracle Cloud Infrastructure**.

**Live API**: http://138.2.180.250:8080

```
GET  /health          → {"status": "healthy"}
GET  /                → {"status": "ok", "model": "rf"}
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
Open `notebooks/01-compare-models.ipynb` in VS Code or Jupyter and run all cells. This trains and compares RF, XGBoost, and GRU models, selects the best (Random Forest), and saves artifacts to `models/`. Use `notebooks/02-bitcoin-price-prediction.ipynb` for deeper EDA and feature exploration.

### 5. View MLflow experiments
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/mlruns
```
Then open http://localhost:5000 in your browser.

---

## How It All Fits Together

The project has three distinct phases — **model selection** (done once, locally), **initial deployment** (done once, manually), and **daily automated retraining** (ongoing, fully automated).

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — Model Selection (local, one-time)                                ║
║                                                                              ║
║  notebooks/01-compare-models.ipynb                                           ║
║  ├─ Download BTC-USD history (yfinance)                                      ║
║  ├─ Engineer 25 features (lags, RSI, MACD, std30, return)                    ║
║  ├─ Train & evaluate: LinearRegression / RandomForest / XGBoost / GRU        ║
║  ├─ Track experiments with MLflow (local SQLite)                             ║
║  └─ Save winner → models/rf_model.save + scaler_X.pkl + scaler_y.pkl        ║
║                           │                                                  ║
╚═══════════════════════════╪══════════════════════════════════════════════════╝
                            │
                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — Initial Deployment (local → cloud, one-time)                     ║
║                                                                              ║
║  docker build -t brunopulheze/btc-predictor:latest .                        ║
║      └─ Dockerfile: COPY models/ /app/models/   ← model baked into image    ║
║  docker push brunopulheze/btc-predictor:latest  → Docker Hub                ║
║                                                                              ║
║  SSH → OCI VM (VM.Standard.A1.Flex, ARM, Frankfurt)                         ║
║      docker pull brunopulheze/btc-predictor:latest                          ║
║      docker run -d -p 8080:8080 --name btc-predictor ...                    ║
║      └─ FastAPI serves /predict/latest, /drift-report, /health              ║
║                                                                              ║
║  Vercel: auto-deploys dashboard/ on every push to main                      ║
║      └─ Next.js dashboard calls OCI API every 30 min (ISR)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                            │
                            ▼ (every day at 06:00 UTC)
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 3 — Daily Automated Retraining (GitHub Actions, ephemeral runner)    ║
║                                                                              ║
║  .github/workflows/retrain.yml                                               ║
║  ├─ git clone → fetch latest model artifacts from repo                       ║
║  ├─ python src/training/retrain.py                                           ║
║  │   ├─ Download fresh BTC-USD history                                       ║
║  │   ├─ Run current model on last 30 days → recent MAE                       ║
║  │   ├─ Drift check: recent_MAE > 1.5 × baseline_RMSE ?                     ║
║  │   │                                                                       ║
║  │   ├─ NO DRIFT → skip retrain                                              ║
║  │   │   └─ write drift_report.json (retrained: false)                       ║
║  │   │                                                                       ║
║  │   └─ DRIFT (or --force) → retrain RF on full history                      ║
║  │       ├─ overwrite rf_model.save / scaler_*.pkl                           ║
║  │       ├─ update selection.json (new RMSE)                                 ║
║  │       ├─ write drift_report.json (retrained: true, new_rmse: ...)         ║
║  │       ├─ docker build → new image with updated model baked in             ║
║  │       ├─ docker push → Docker Hub                                         ║
║  │       └─ SSH → OCI VM: docker pull + stop + rm + run                      ║
║  │                                                                           ║
║  └─ Always: scp drift_report.json → OCI VM → docker cp → container          ║
║      └─ dashboard /drift-report endpoint always shows latest metrics         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Why GitHub Actions for retraining (not the OCI VM)?

| Concern | GitHub Actions | OCI VM |
|---------|---------------|--------|
| CPU | 2-core x86, dedicated per job | 4-core ARM, shared with live API |
| Isolation | Ephemeral — crash can't affect serving | Retrain crash = API crash |
| Cost | Free (2,000 min/month included) | Free — but steals resources from API |
| Rollback | Bad model = revert git commit | Bad model = manual SSH fix |
| Secrets | Managed by GHA (Docker Hub, SSH key) | Would need separate secret management |

The OCI VM is kept lean — it only runs the pre-built Docker image and serves predictions. All heavy computation stays in the ephemeral GHA runner.

See [`docs/retrain.md`](docs/retrain.md) for a full walkthrough of the retraining logic and [`notebooks/04-retrain-pipeline.ipynb`](notebooks/04-retrain-pipeline.ipynb) for an interactive version.

---

## ML Pipeline

| Step | Description |
|------|-------------|
| Data retrieval | Daily BTC/USD prices via `yfinance` from 2015 to today |
| Feature engineering | 20-day lag window, RSI-14, MACD, 30-day rolling std, yesterday's return — all shifted by 1 day (leak-free) |
| Preprocessing | MinMaxScaler on features, StandardScaler on log-return target, 70/30 train/test split |
| Model | **RandomForestRegressor(n_estimators=300)** trained on log-returns |
| Tracking | MLflow logs parameters, RMSE, and artifacts |
| Evaluation | Price space RMSE + log-return space R² & directional accuracy, benchmarked against a persistence baseline |

---

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1) |
| Feature set | 20-day lag window + RSI-14 + MACD + MACD signal + rolling std(30) + yesterday's return (25 features total) |
| Target | Log-return `log(close[t]/close[t-1])` — reconstructed to price at serving time |
| Test RMSE | ~$1,910 (price space, holdout test set) |
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

The Docker image `brunopulheze/btc-predictor:latest` is hosted on Docker Hub and deployed on **Oracle Cloud Infrastructure** — an Always Free `VM.Standard.A1.Flex` ARM instance in Frankfurt (`eu-frankfurt-1`).

| Guide | Platform | URL |
|-------|----------|-----|
| [`docs/deploy-oracle.md`](docs/deploy-oracle.md) | Oracle Cloud VM | http://138.2.180.250:8080 |

---

## Requirements

See `requirements.txt`. Key libraries:

- `scikit-learn` — Random Forest model, preprocessing and metrics
- `yfinance` — Bitcoin price data
- `mlflow` — experiment tracking (local, optional)
- `fastapi` + `uvicorn` — inference API
- `joblib` — model and scaler serialisation
- `tensorflow` + `keras` — used in notebooks for GRU comparison only, not in production

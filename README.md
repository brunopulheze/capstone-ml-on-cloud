# Capstone 2 — Bitcoin Price Prediction (ML on Cloud)

## Overview
End-to-end machine learning project that predicts daily **Bitcoin/USD (BTC-USD)** prices by comparing three models — Linear Regression, Random Forest, and a stacked LSTM neural network. Experiment tracking is handled by MLflow and the project follows an MLOps deployment roadmap on AWS.

## Project Structure

```
capstone-ml-on-cloud/
├── notebooks/
│   └── bitcoin-price-prediction.ipynb     # Main notebook (EDA → 3 models → comparison)
├── models/
│   ├── keras_model.h5                     # Trained LSTM model
│   └── scaler.save                        # Fitted MinMaxScaler
├── src/
│   ├── data/                              # Data pipeline scripts
│   ├── training/                          # Model training scripts
│   └── api/                              # FastAPI inference service
├── tests/                                 # Unit tests
├── docs/
│   └── capstone-briefing.md               # Project briefing
├── mlflow/
│   └── mlruns/                            # MLflow experiment artifacts (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

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
Open `notebooks/bitcoin-price-prediction.ipynb` in VS Code or Jupyter and run all cells.

### 5. View MLflow experiments
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/mlruns
```
Then open http://localhost:5000 in your browser.

## ML Pipeline

| Step | Description |
|------|-------------|
| Data retrieval | Daily Bitcoin/USD prices (`BTC-USD`) via `yfinance` from 2015 to today |
| EDA | Shape, missing values, descriptive stats, close price & moving average plots |
| Preprocessing | MinMax scaling + 100-step sliding window sequences (70/30 train/test split) |
| Model A | **Linear Regression** — flat 100-feature baseline |
| Model B | **Random Forest** — 100-tree ensemble, captures non-linear patterns |
| Model C | **LSTM** — 4-layer stacked network (50→60→80→120 units), EarlyStopping |
| Tracking | MLflow logs all three models under `bitcoin-price-prediction` experiment |
| Evaluation | MAE, RMSE, R² comparison table + all-models line plot + R² bar chart |

## Model Comparison

All three models are evaluated on the same held-out 30% test set:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| Linear Regression | Fast, interpretable | Assumes linearity, no temporal context |
| Random Forest | Handles non-linearity, robust to noise | No native sequence awareness |
| LSTM | Native temporal memory, learns long-range patterns | Requires more data and tuning |

## Deployment Roadmap (MLOps)

- **Tier 1** ✅ — Git repo, virtual environment, `requirements.txt`
- **Tier 2** — MLflow tracking ✅, AWS deployment (SageMaker / EC2), FastAPI REST endpoint
- **Tier 3** — Docker containerisation, AWS ECR + ECS
- **Tier 4** — CI/CD with GitHub Actions

## Requirements

See `requirements.txt`. Key libraries:
- `tensorflow` — LSTM model
- `scikit-learn` — Linear Regression, Random Forest, preprocessing and metrics
- `yfinance` — Bitcoin price data
- `mlflow` — experiment tracking
- `fastapi` + `uvicorn` — inference API (deployment)
- `boto3` — AWS SDK

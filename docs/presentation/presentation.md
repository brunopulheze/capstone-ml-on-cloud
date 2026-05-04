---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    background: #f8f7ff;
    color: #2a2554;
  }
  h1 { color: #3b82f6; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }
  h2 { color: #2a2554; }
  table { font-size: 0.85em; }
  th { background: #3b82f6; color: white; }
  code { background: #e8e4ff; border-radius: 4px; padding: 2px 5px; }
  .highlight { color: #3b82f6; font-weight: bold; }
---

# BTC Price Predictor
## End-to-End ML on the Cloud

**Capstone Project — Ironhack Data Science & AI**

Bruno Pulheze · April 2026

> From raw data on Yahoo Finance to a live dashboard on Vercel,
> with automated retraining, drift monitoring, and Oracle Cloud deployment.

---

## Slide 1 — What We Built

**Goal:** Predict the next-day Bitcoin closing price and serve it as a live API.

### Full Stack

| Layer | Technology |
|---|---|
| Data | Yahoo Finance via `yfinance` (~4,200 daily closes since 2014) |
| Training | scikit-learn · XGBoost · Keras GRU (comparison) |
| Experiment tracking | **MLflow** — SQLite backend |
| API | **FastAPI** + uvicorn |
| Container | **Docker** → Docker Hub |
| Cloud VM | **Oracle Cloud Infrastructure** — Always Free ARM instance |
| CI/CD + retraining | **GitHub Actions** — daily cron + manual trigger |
| Dashboard | **Next.js 15** + Recharts → deployed on **Vercel** |

---

## Slide 2 — Lesson Learned: Price-Space Was a Trap

### First approach — predict the raw closing price

A model that simply outputs **"tomorrow ≈ today"** achieves an R² of ~0.99 in price-space.
This is not skill — it is just persistence. The numbers look great; the model is useless.

```
Persistence:  price[t+1] = price[t]   →  price-space RMSE ≈ $800
A model that predicts "no change" every day can match this trivially.
```

### The fix — predict log-returns instead

$$\text{target} = \log\!\left(\frac{\text{close}[t]}{\text{close}[t-1]}\right)$$

- Centered around **zero** — the model must learn real signal, not just copy yesterday's price
- **Stationary** — no extrapolation failures when Bitcoin reaches new all-time highs
- **Honest baseline** — persistence predicts log-return = 0 every day (RMSE ≈ σ of returns ≈ 0.038)
- Price is recovered at serving time: `price[t+1] = price[t] × exp(predicted_return)`

> Any model that cannot beat the persistence baseline in log-return space has learned nothing useful.

---

## Slide 3 — Model Comparison

Three candidates trained on **25 features**, evaluated in **log-return space**.

### Features (25 total — all shifted by 1 day, no lookahead)

`lag_1 … lag_20` (closing prices) · `std30` · `rsi14` · `macd` · `macd_sig` · `return`

### Holdout results (70/30 split)

| Model | RMSE | MAE | R² | Dir Acc |
|---|---|---|---|---|
| RF | 0.03660 | 0.02908 | −1.17 | 52.8% |
| XGBoost | 0.03059 | 0.02296 | −0.52 | 49.7% |
| GRU | 0.02542 | 0.01808 | −0.05 | 49.4% |
| Persistence baseline | ~0.038 | — | 0 | 50% |

### Walk-forward CV (200 steps, trees refit every 7 steps — more realistic)

| Model | RMSE | MAE | Dir Acc |
|---|---|---|---|
| **RF** | **0.00159** | **0.00088** | **95.5%** ✓ |
| GRU | 0.00197 | 0.00177 | 32.5% |
| XGBoost | 0.00502 | 0.00142 | 93.5% |

**Random Forest** wins walk-forward CV: lowest RMSE and highest directional accuracy.
GRU had better holdout metrics but failed to generalize in the sequential simulation.

---

## Slide 4 — MLflow & Model Selection

### Experiment tracking with MLflow

Every training run logs:
- Hyperparameters: `n_estimators`, feature list, train/test split date
- Metrics: `logret_rmse`, `logret_mae`, `R²`, `dir_acc`
- Artifacts: model file, scalers

```bash
mlflow ui   # SQLite backend at mlflow/mlflow.db → http://localhost:5000
```

### Selection logic (`src/training/compare_models.py`)

The best model is chosen by **walk-forward log-return RMSE** (not price-space RMSE).
Result written to `models/selection.json` — the single source of truth for the API.

```json
{
  "model_type": "rf",
  "logret_rmse": 0.02787,
  "rmse": 1917.52,
  "persistence_rmse": 1716.76,
  "features": ["lag_1", ..., "lag_20", "std30", "rsi14", "macd", "macd_sig", "return"]
}
```

> `selection.json` decouples training from serving: the API loads whatever model
> is recorded there without any code change.

---

## Slide 5 — Automated Retraining & Drift Monitoring

### Daily GitHub Actions cron (`06:00 UTC`)

```
1. Download BTC-USD history (yfinance)
2. Build feature matrix — same 25 features as training
3. Load existing RF model + scalers
4. Evaluate on last 30 days  →  recent MAE
5. Drift check: recent MAE > 1.5 × baseline RMSE?
   └─ YES → retrain from scratch, save new artifacts
   └─ NO  → log "no drift", exit
6. Write models/drift_report.json  →  dashboard reads this live
```

### Drift report (visible on dashboard)

```json
{
  "available": true,
  "drifted": false,
  "recent_mae": 1423.50,
  "baseline_rmse": 1716.76,
  "drift_threshold": 2575.14
}
```

Force retrain anytime:

```bash
python src/training/retrain.py --force
# or: GitHub Actions → Run workflow → force_retrain=true
```

---

## Slide 6 — Deployment: Docker → Oracle Cloud → Vercel

### Container (`Dockerfile`)

```
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/       ← RF artifacts baked in at build time
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t brunopulheze/btc-predictor:latest .
docker push brunopulheze/btc-predictor:latest
```

### Oracle Cloud Infrastructure — Always Free tier

- **Instance:** `VM.Standard.A1.Flex` — ARM Ampere, 1 OCPU, 6 GB RAM, permanently free
- **Live API:** `http://138.2.180.250:8080`
- Endpoints: `GET /`, `GET /health`, `GET /predict/latest`, `POST /predict`

### Smoke test (contract test before every deploy)

```bash
python tests/smoke_test.py --url http://138.2.180.250:8080
# checks: GET / → model=RF · POST /predict → predicted_price is float > 0
```

### Vercel dashboard

- **Next.js 15** App Router — server component fetches API + CoinGecko in parallel
- **ISR** (revalidate = 1800 s) — fresh data every 30 min without full rebuild
- Displays: 60-day price chart · next-day prediction · drift monitoring card · model metadata

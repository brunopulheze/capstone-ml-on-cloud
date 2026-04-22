# Codebase Guide — Scripts & Tests

This document explains every runnable script in the project: what it does, when to use it, and how to run it.

---

## `src/training/retrain.py` — Drift Detection & Retraining

### Purpose
Automated pipeline that detects whether the live GRU model has degraded in performance ("drift") on recent data, and retrains it if so. This is the core of the **Tier 2** model monitoring requirement.

### How it works

```
1. Download full BTC-USD history (yfinance, 2014 → today)
2. Build features  ← identical pipeline to app.py (lag_1…lag_100, RSI-14, MACD, std30, return)
3. Load current model (best_model.keras) + scalers from models/
4. Predict the last 30 days   →   compute recent MAE
5. Compare: recent MAE  vs.  1.5 × baseline RMSE (from selection.json)
      ├─ NO drift  →  write drift_report.json, exit
      └─ DRIFT     →  retrain GRU from scratch (70/30 split)
                       save best_model.keras, scaler_X.pkl, scaler_y.pkl, selection.json
                       log run to MLflow (if installed)
                       write drift_report.json
```

### Drift threshold

| Variable | Default | Meaning |
|----------|---------|---------|
| `EVAL_DAYS` | 30 | Days of recent predictions to evaluate |
| `DRIFT_THRESHOLD` | 1.5 | Retrain if recent MAE > 1.5 × baseline RMSE |

With baseline RMSE ≈ $1,901, retraining triggers when the model's MAE over the last 30 days exceeds **~$2,850**.

### Output: `models/drift_report.json`
Always written after a run:
```json
{
  "timestamp":       "2026-04-22T06:00:01",
  "eval_days":       30,
  "recent_mae":      1823.45,
  "baseline_rmse":   1901.23,
  "drift_threshold": 2851.85,
  "drift_detected":  false,
  "retrained":       false
}
```
If retraining occurred, `retrained: true` and `new_rmse` are added.

### Usage

```bash
# Normal run (retrain only if drift detected)
python src/training/retrain.py

# Force retrain regardless of performance
python src/training/retrain.py --force
```

### Dependencies
Requires the same `requirements.txt` environment. Does **not** need the API server to be running.

---

## `.github/workflows/retrain.yml` — Daily GitHub Actions Pipeline

### Purpose
Runs `retrain.py` automatically every day at **06:00 UTC**. If the model is retrained, it also rebuilds the Docker image, pushes it to Docker Hub, and redeploys the container on the Oracle Cloud VM — fully hands-free.

### Trigger conditions

| Trigger | When |
|---------|------|
| **Scheduled** | Every day at 06:00 UTC (`cron: '0 6 * * *'`) |
| **Manual** | GitHub Actions → Run workflow → optionally check `force_retrain` |

### Job steps

```
1. Checkout repository
2. Set up Python 3.11 + install requirements.txt
3. Run: python src/training/retrain.py [--force]
4. Read models/drift_report.json → set output "retrained=true/false"
5. Upload drift_report.json as a workflow artifact (always)

If retrained=true:
  6. git commit updated model artifacts → git push
  7. docker build + push brunopulheze/btc-predictor:latest
  8. SSH into Oracle Cloud VM → docker pull + restart container
```

### Required GitHub Secrets

Go to **GitHub repo → Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | `brunopulheze` |
| `DOCKERHUB_TOKEN` | Docker Hub personal access token (Hub → Account Settings → Personal access tokens) |
| `OCI_HOST` | `138.2.180.250` |
| `OCI_SSH_KEY` | Full contents of `~/.ssh/oci-btc-key` (including `-----BEGIN/END OPENSSH PRIVATE KEY-----` lines) |

### Manual trigger (force retrain)
1. GitHub → **Actions** tab → **Daily BTC Drift Check & Retrain**
2. Click **Run workflow**
3. Set `force_retrain` to `true` → **Run workflow**

### Artifacts
Each run uploads `drift_report.json` under **Actions → run → Artifacts** for auditing.

---

## `tests/smoke_test.py` — API Smoke Test

### Purpose
End-to-end test that verifies the API is reachable and returning correct predictions. Runs against any deployment target: local, Docker, or the live Oracle Cloud instance.

### What it tests

| Check | Endpoint |
|-------|----------|
| Server is reachable | `GET /` |
| Health check passes | `GET /health` |
| Prediction returns valid data | `POST /predict` (with full BTC-USD history from yfinance) |

Assertions on `POST /predict` response:
- `predicted_price` is a positive float
- `previous_close` is a positive float

### Usage

```powershell
# Against Oracle Cloud (live)
python tests/smoke_test.py --url http://138.2.180.250:8080

# Against local Docker container
python tests/smoke_test.py --port 8081

# Against local uvicorn
python tests/smoke_test.py --port 8080
```

Exit code `0` = all passed, `1` = failure (with error message on stderr).

---

## `src/training/compare_models.py` — Model Comparison (historical, not automated)

### Purpose
One-off training script used during the **model selection phase**. Trains RandomForest, XGBoost, and LSTM on the same engineered features and evaluates each with both hold-out and walk-forward cross-validation. Writes `models/selection.json` with the winning model.

> This script is **not part of the automated pipeline**. It was used in the notebook exploration phase to decide that GRU outperformed all three. The GRU training is now handled exclusively by `retrain.py` and the main notebook.

### Feature set (used for comparison)
- 100-day lag window (`lag_1` … `lag_100`)
- 100-day rolling mean (shifted)
- Rolling std(30) (shifted)
- RSI-14, MACD, MACD signal (all shifted by 1 day)
- Daily return

### Models compared

| Model | Notes |
|-------|-------|
| `RandomForest` | 100 estimators, hold-out + walk-forward |
| `XGBoost` | 100 estimators, hold-out + walk-forward |
| `LSTM(64)` | 3 epochs (quick comparison, not tuned); trained once for walk-forward |

### Usage (manual only)

```bash
python src/training/compare_models.py
```

Outputs metrics to stdout and saves `models/selection.json`.

> **Note**: the final GRU model is trained separately in `notebooks/bitcoin-price-prediction.ipynb`, which supersedes this script's model artifacts.

---

## `src/training/compare_lr_features.py` — ~~Dead Code~~ (deleted)

This script compared Linear Regression variants (overlapping vs non-overlapping lag windows, with/without a 100-day rolling average) as an early experiment. It was never imported or called by any other file in the project and was deleted during cleanup.

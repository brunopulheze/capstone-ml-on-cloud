# Automated Retraining Pipeline — `src/training/retrain.py`

## Purpose

`retrain.py` runs every day via GitHub Actions (06:00 UTC). Its job is to check whether the deployed Random Forest model has become less accurate on recent BTC prices — a phenomenon called **model drift** — and retrain it from scratch if needed.

It can also be triggered manually from the command line:

```bash
# Normal run — retrain only if drift is detected
python src/training/retrain.py

# Force a retrain regardless of current model performance
python src/training/retrain.py --force
```

---

## How it works — step by step

```
┌──────────────────────────────────────────────────────────────────┐
│  1. Download BTC-USD history (yfinance, 2014 → today)            │
│  2. Build feature matrix (25 features, same as app.py)          │
│  3. Load existing model + scalers from models/                   │
│     └─ If no model exists → cold start, skip to step 5           │
│  4. Evaluate model on last 30 days → recent MAE                  │
│     └─ MAE > 1.5 × baseline RMSE? → drift = True                 │
│  5. If drift OR --force → retrain RF from scratch                │
│     └─ Save new model artifacts to models/                       │
│  6. Write models/drift_report.json                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Step 1 — Download data

```python
raw = yf.download("BTC-USD", start="2014-01-01", ...)
prices = raw["Close"].dropna().tolist()
```

Downloads the full price history from Yahoo Finance (~4 200 daily closes). Starting from 2014 ensures enough data for the lag window and indicators to warm up without losing rows.

---

## Step 2 — Build features

Calls `build_features(prices)`, which produces the same 25-column feature matrix used during training and inference in `app.py`. Every column is shifted by 1 day so no future information leaks into the features.

| Feature group | Columns | What it captures |
|---|---|---|
| Lag window | `lag_1` … `lag_20` | Raw closing prices for the previous 1–20 days |
| Volatility | `std30` | 30-day rolling standard deviation |
| Momentum | `rsi14` | Relative Strength Index (0–100) |
| Trend | `macd`, `macd_sig` | MACD line and signal line |
| Direction | `return` | Previous day's percentage change |

---

## Step 3 — Load existing model

Reads `models/selection.json` to retrieve the last known RMSE and the model configuration. Then loads:

| File | Contents |
|---|---|
| `models/rf_model.save` | Trained Random Forest (joblib) |
| `models/scaler_X.pkl` | MinMaxScaler fitted on training features |
| `models/scaler_y.pkl` | StandardScaler fitted on training log-returns |

**Cold start:** If `rf_model.save` does not exist (e.g. first run on a fresh clone), this step is skipped and the pipeline goes directly to retraining.

---

## Step 4 — Drift detection

Calls `evaluate_recent(feat_df, model, scaler_X, scaler_y, n=30)`:

1. Takes the last 30 rows of the feature matrix (= last 30 trading days)
2. Runs the full prediction pipeline on each row (same as `app.py`)
3. Compares predicted prices to actual closes → computes MAE in USD

The drift decision:

```
drift = recent_MAE  >  DRIFT_THRESHOLD × baseline_RMSE
                            1.5        ×    ~$622
                    ≈ $933
```

| Result | Meaning |
|---|---|
| `drift = False` | Model is still performing within acceptable bounds — no retrain |
| `drift = True` | Model has degraded significantly — retrain triggered |

---

## Step 5 — Retraining (only if needed)

Calls `retrain(feat_df)`, which trains a brand-new Random Forest from scratch on the full price history:

1. **Split** — chronological 70/30 (no shuffling, to avoid future leakage)
2. **Scale** — fit MinMaxScaler on X_train only; fit StandardScaler on y_train only
3. **Train** — `RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)`
4. **Evaluate** — compute RMSE on the held-out test set in USD (after back-transforming predictions)
5. **Save** — write `rf_model.save`, `scaler_X.pkl`, `scaler_y.pkl`, and update `selection.json`

The back-transformation from predicted log-return to USD price:

$$\text{price}[t] = \text{close}[t-1] \times e^{\hat{y}}$$

---

## Step 6 — Drift report

Always writes `models/drift_report.json` regardless of whether a retrain happened:

```json
{
  "timestamp":       "2026-04-26T06:12:34.000000",
  "eval_days":       30,
  "recent_mae":      2103.45,
  "baseline_rmse":   1901.23,
  "drift_threshold": 2851.85,
  "drift_detected":  false,
  "retrained":       false
}
```

If retrained, two extra fields appear:

```json
{
  "retrained": true,
  "new_rmse":  1784.60
}
```

This file is consumed by the FastAPI `GET /drift-report` endpoint and displayed in the dashboard.

---

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `SEQ_LEN` | 20 | Number of lag features |
| `EVAL_DAYS` | 30 | Recent window used for drift detection |
| `DRIFT_THRESHOLD` | 1.5 | Multiplier applied to baseline RMSE to set the drift threshold |

---

## Functions

| Function | Purpose |
|---|---|
| `_rsi(series, period)` | Computes RSI — a momentum oscillator bounded 0–100 |
| `_macd(series, a, b, c)` | Computes MACD line and signal line from three EMAs |
| `build_features(prices)` | Builds the full 25-column feature matrix from raw closes |
| `_feature_cols(seq_len)` | Returns the ordered list of column names the model expects |
| `evaluate_recent(...)` | Runs the current model on the last 30 days and returns MAE |
| `retrain(feat_df, ...)` | Trains a new RF from scratch and returns model + scalers + RMSE |
| `main(force)` | Orchestrates all steps; writes drift report; entry point |
| `_NullContext` | Dummy context manager used when MLflow is not installed |

---

## Outputs

| File | Written when |
|---|---|
| `models/drift_report.json` | Every run |
| `models/rf_model.save` | Retrain only |
| `models/scaler_X.pkl` | Retrain only |
| `models/scaler_y.pkl` | Retrain only |
| `models/selection.json` | Retrain only (RMSE and timestamp updated) |

---

## GitHub Actions integration

The pipeline is wired to `.github/workflows/retrain.yml`:

1. Checks out the repo (model artifacts are now committed, so the runner has them)
2. Installs dependencies from `requirements.txt`
3. Runs `retrain.py` (with `--force` if the `force_retrain` input is `true`)
4. Reads `drift_report.json` to decide whether downstream steps should run
5. If `retrained = true`: commits updated artifacts → builds and pushes a new Docker image → redeploys on Oracle Cloud

See [`deploy-ec2.md`](../deploy-ec2.md) for the full deployment guide.

---

## MLflow (optional)

If `mlflow` is installed, each retrain run is logged to the `bitcoin-price-prediction` experiment with:

| Logged item | Value |
|---|---|
| Param: `trigger` | `"forced"` or `"drift detected"` |
| Param: `seq_len` | 20 |
| Param: `eval_days` | 30 |
| Metric: `pre_retrain_mae` | MAE before retraining (if model existed) |
| Metric: `new_rmse` | RMSE of the newly trained model |

If MLflow is not installed, a `_NullContext` is used instead and the run proceeds silently without logging.

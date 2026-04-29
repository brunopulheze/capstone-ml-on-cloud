# BTC Price Predictor — Presentation Notes

**Capstone Project · Ironhack Data Science & AI · April 2026**
**Bruno Pulheze**

---

## Introduction

The goal of this capstone was to build an end-to-end machine learning product with real-world deployment: a Bitcoin next-day price predictor that runs live on the cloud, updates automatically, and presents its output through a public dashboard.

The project covers the full MLOps cycle — data collection, feature engineering, model comparison, experiment tracking with MLflow, containerization with Docker, cloud deployment on Oracle Cloud Infrastructure, automated retraining triggered by drift detection, and a live Next.js dashboard deployed on Vercel.

The asset chosen was Bitcoin/USD. While the original brief described Gold/USD on AWS, the core requirements are identical: a financial time series, a deployed prediction endpoint, MLOps automation, and a CI/CD pipeline. Oracle Cloud's Always Free tier was used instead of AWS, which provides a permanently free ARM virtual machine with enough compute for this workload.

---

## 1. The Problem with Predicting Prices Directly

The first instinct when building a price predictor is to train a model to output the next closing price. This seems natural — the target is a price, so predict a price. In practice, this approach produces numbers that look excellent on paper but are completely misleading.

The reason is that Bitcoin's price tomorrow is almost always very close to Bitcoin's price today. A model that simply outputs "tomorrow = today" — the persistence baseline — achieves an R² close to 0.99 and a very low RMSE in absolute dollar terms. Any model that learns from the data will pick up this pattern and essentially learn to copy the most recent price. The evaluation metrics reward this behaviour, not genuine forecasting skill.

The fix is to change the prediction target to **log-returns**: the natural logarithm of the ratio between consecutive closing prices. This quantity is stationary — it fluctuates around zero regardless of whether Bitcoin is priced at $10,000 or $100,000 — and a persistence baseline that always predicts zero has a well-defined RMSE equal to the standard deviation of daily returns (approximately 0.038). Any model that cannot beat this has learned nothing of value.

At serving time, the predicted return is converted back to a price: `price[t+1] = price[t] × exp(predicted_return)`. The user sees a dollar figure, but the model and all evaluation metrics operate in return space.

---

## 2. Feature Engineering

All features are derived from the raw daily closing price history downloaded from Yahoo Finance via `yfinance`, covering approximately 4,200 trading days from January 2014 to the present.

The feature set contains 25 columns per row:

- **Lag window** (`lag_1` through `lag_20`): the closing prices for each of the previous 20 trading days. These give the model a direct view of recent price history.
- **Volatility** (`std30`): the 30-day rolling standard deviation of the closing price, used as a proxy for market turbulence.
- **Momentum** (`rsi14`): the Relative Strength Index over 14 days, a bounded oscillator between 0 and 100 that signals overbought or oversold conditions.
- **Trend** (`macd`, `macd_sig`): the MACD line (the difference between the 12-day and 26-day exponential moving averages) and its 9-day signal line.
- **Direction** (`return`): the previous day's percentage change in the closing price.

Every feature is shifted by one day before being used as input, ensuring no information from day `t` leaks into a prediction about day `t`. This is the most common source of data leakage in financial ML projects and was treated carefully throughout.

---

## 3. Model Comparison

Three candidate models were trained and evaluated: Random Forest, XGBoost, and a GRU (Gated Recurrent Unit) neural network.

**Random Forest** is a bagging ensemble of decision trees. Each tree is trained on a random bootstrap sample and uses a random subset of features at each split. The final prediction is the average across all trees. It handles non-linear relationships well and is robust to outliers.

**XGBoost** uses gradient boosting — trees are built sequentially, each one correcting the residuals of the previous. It generally outperforms Random Forest on tabular data with appropriate tuning.

**GRU** is a recurrent neural network designed for sequential data. It processes the 20 most recent closing prices as an ordered sequence and uses gating mechanisms to selectively retain or discard information across timesteps.

Two evaluation protocols were used, both operating in log-return space:

The **70/30 holdout** split showed GRU with the best RMSE (0.025) and RF with the worst (0.037), which initially favoured the neural network.

The **walk-forward cross-validation** is a more realistic simulation: the model is evaluated step-by-step on the test set, with tree models periodically refitting on all available past data as the window grows. Under this protocol, RF achieved an RMSE of 0.00159 and 95.5% directional accuracy, outperforming GRU (RMSE 0.00197, 32.5% directional accuracy). GRU's directional accuracy collapsed to near-random in the walk-forward setting, suggesting it had learned patterns specific to the training period rather than generalizable signal.

Random Forest was selected as the production model based on walk-forward performance, which is the more conservative and operationally honest metric.

---

## 4. Experiment Tracking with MLflow

Every training run — both in the comparison notebook and in the automated retraining pipeline — is logged to MLflow. The backend is a local SQLite database at `mlflow/mlflow.db`, which can be browsed with `mlflow ui` on port 5000.

Each run records the hyperparameters (`n_estimators`, feature list, train/test split), the metrics (`logret_rmse`, `logret_mae`, R², directional accuracy), and the saved model artifacts.

Model selection is performed by comparing walk-forward log-return RMSE across all logged runs. The result is written to `models/selection.json`, which acts as the single source of truth for the API. The serving code reads this file at startup and loads whatever model is recorded there — no code change is needed when a new model is promoted to production.

---

## 5. Serving the Model: FastAPI + Docker

The prediction endpoint is built with FastAPI and exposes four routes:

- `GET /` returns the server status and confirms the loaded model type and feature count.
- `GET /health` is a lightweight liveness check used by monitoring tools.
- `GET /predict/latest` is the autonomous endpoint used by the dashboard. It downloads the full BTC-USD history internally via `yfinance`, constructs the feature vector, and returns tomorrow's predicted price along with the previous close and a data quality summary — no input from the caller required.
- `POST /predict` accepts a list of closing prices from the caller and runs the same pipeline.

The application is containerized with Docker. The Dockerfile uses a Python 3.11 slim base image, installs dependencies from `requirements.txt`, copies the source and model artifacts, and starts uvicorn on port 8080. The image is published to Docker Hub as `brunopulheze/btc-predictor:latest`.

A smoke test in `tests/smoke_test.py` validates the API contract before every deploy: it calls the root and health endpoints, sends a real price history to `/predict`, and asserts that the response contains a `predicted_price` and `previous_close` that are both positive floats.

---

## 6. Cloud Deployment: Oracle Cloud Infrastructure

The container runs on an Oracle Cloud Infrastructure Always Free virtual machine — a `VM.Standard.A1.Flex` instance using the ARM Ampere architecture, with 1 OCPU and 6 GB of RAM, permanently free of charge.

The deployment process involved provisioning the VM through the OCI console, configuring the network security list to expose port 8080 to the internet, opening the port in the OS-level firewall with `iptables`, installing Docker on the Ubuntu instance, and pulling and starting the container.

The live API is reachable at `http://138.2.180.250:8080` and has been running continuously since initial deployment.

---

## 7. Automated Retraining and Drift Detection

A GitHub Actions workflow runs every day at 06:00 UTC. It downloads the latest BTC-USD history, evaluates the deployed model on the most recent 30 trading days, and compares the resulting MAE against a drift threshold set at 1.5 times the baseline RMSE recorded at training time.

If the model's recent error exceeds the threshold — indicating that the distribution of returns has shifted and the model is no longer accurate — the pipeline retrains a new Random Forest from scratch, saves the updated artifacts to the `models/` directory, and logs the run to MLflow. If no drift is detected, the run exits cleanly without retraining.

The drift report is written to `models/drift_report.json` after every run and is served through the API's `/drift-report` endpoint. The dashboard reads this live and displays the current drift status, recent MAE, baseline RMSE, and threshold — giving operators visibility into model health without needing to inspect logs or the MLflow UI directly.

The workflow also supports manual triggering with a `force_retrain=true` parameter, which bypasses the drift check and retrains unconditionally. This is useful after intentional changes to the feature set or model hyperparameters.

---

## 8. The Dashboard

The user-facing interface is a single-page application built with Next.js 15, Recharts, and Tailwind CSS, deployed on Vercel.

The server component fetches data in parallel at build and revalidation time: the prediction and drift report from the Oracle Cloud API, and the 90-day price history from the CoinGecko public API. Vercel's Incremental Static Regeneration is configured with a 30-minute window, meaning pages are refreshed from live data at most twice per hour without a full rebuild.

The dashboard displays a 60-day price chart with the next-day prediction plotted as a dashed extension, four stat cards showing tomorrow's predicted price, current live price, previous close, and predicted percentage change, a model architecture card with the Random Forest configuration and test RMSE, and a drift monitoring card showing the latest report from the retraining pipeline.

Every push to the `main` branch triggers an automatic Vercel redeploy, so documentation and dashboard updates go live within about one minute of being merged.

---

## Summary

The project demonstrates a complete MLOps lifecycle applied to a real financial forecasting problem:

1. **Honest evaluation** — switching from price-space to log-return space, and from holdout to walk-forward CV, exposed that GRU's apparent advantage was not generalizable.
2. **Experiment tracking** — MLflow captured every run and made model selection reproducible and auditable.
3. **Clean serving layer** — the API decouples the model from the calling code via `selection.json`; promoting a new model requires no code changes.
4. **Containerization** — Docker ensures the same environment runs locally, in CI, and on the cloud VM.
5. **Automated operations** — drift detection and retraining run daily without human intervention; the dashboard surfaces model health in real time.
6. **Live product** — the Oracle Cloud API and Vercel dashboard are both publicly accessible and have been running continuously.

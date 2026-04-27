# Bitcoin Price Prediction — Model Overview

## Goal

Predict the next-day Bitcoin closing price using three machine learning models trained on historical BTC-USD data (2015–present). Models predict the **log-return** `log(close[t] / close[t-1])`, which is then back-transformed to a price for evaluation.

---

## Models

### Random Forest (RF)
An ensemble of decision trees that each vote on a prediction. Each tree is trained on a random bootstrap sample of the data and considers a random subset of features at each split, reducing variance.

- **Type:** Ensemble — bagging
- **Key hyperparameters:** `n_estimators=100`, `n_jobs=-1` (parallel)
- **Strengths:** Handles non-linear relationships, robust to outliers, no scaling required (scaling applied anyway for consistency)
- **Limitation:** Cannot extrapolate beyond the price range seen during training — addressed by predicting log-returns instead of raw prices

### XGBoost
Gradient-boosted decision trees that iteratively correct the residuals of previous trees. Generally more accurate than RF on tabular data with proper tuning.

- **Type:** Ensemble — gradient boosting
- **Key hyperparameters:** `n_estimators=100`, `n_jobs=-1`
- **Strengths:** Strong out-of-the-box performance, handles sparse features, fast training
- **Limitation:** Same extrapolation issue as RF — resolved via log-return target

### GRU (Gated Recurrent Unit)
A recurrent neural network architecture designed for sequential data. It uses gating mechanisms to selectively remember or forget information across timesteps, capturing temporal dependencies more efficiently than a vanilla RNN.

- **Type:** Deep learning — recurrent sequence model
- **Architecture:** `GRU(64 units)` → `Dense(1)`
- **Input:** Last `LOOKBACK=20` days of normalized closing prices, ordered chronologically — shape `(n, 20, 1)`
- **Training:** Adam optimizer, MSE loss, 3 epochs, batch size 32
- **Strengths:** Learns temporal patterns from price sequences; less prone to extrapolation failure than tree models
- **Limitation:** Can learn persistence (`price[t+1] ≈ price[t]`) rather than genuine signal — the log-return target mitigates this by centering predictions around zero

---

## Features

All features are computed from the raw `Close` price and **shifted by 1 day** to prevent lookahead leakage (only information available before day `t` is used to predict day `t`).

| Feature | Description |
|---|---|
| `lag_1` … `lag_100` | Closing prices for the previous 1–100 trading days |
| `std30` | 30-day rolling standard deviation of Close (volatility proxy) |
| `rsi14` | Relative Strength Index over 14 days (momentum oscillator, 0–100) |
| `macd` | MACD line: EMA(12) − EMA(26) of Close (trend indicator) |
| `macd_sig` | MACD signal line: EMA(9) of the MACD line |
| `return` | Previous day's percentage change in Close |

**Total features:** 105

### Scaling
- **X (features):** `MinMaxScaler` — fits to `[0, 1]` based on training range
- **y (target):** `StandardScaler` — standardizes log-returns to zero mean and unit variance; handles negative values correctly

---

## How models use the full price history

All three models are trained on the full BTC-USD history downloaded from Yahoo Finance (~4 200 daily closes from 2014 to today). The raw history is converted into a feature matrix where **each row represents one trading day** and contains all 105 features for that day. This gives ~4 100 training rows (some early rows are dropped during indicator warm-up).

Every model trains on all ~4 100 rows. The difference is in what each row looks like to the model.

### Trees — flat feature vector (105 numbers, no order)

For Random Forest and XGBoost, each row is presented as a flat list of 105 independent numbers:

```
Row for day t:
[lag_1, lag_2, lag_3, ..., lag_100, std30, rsi14, macd, macd_sig, return]
  ↑ yesterday          ↑ 100 days ago
```

The model has no concept of which lag came before which. It just learns rules like "when `lag_1` is high *and* `rsi14` > 70, the return tends to be negative" — treating all 105 columns as simultaneous inputs with no temporal relationship between them.

### GRU — ordered sequence (20 timesteps, time matters)

For the GRU, each row's first 20 lag columns are extracted and **arranged in chronological order** as a sequence of shape `(20, 1)`:

```
Row for day t, reshaped for GRU:
timestep:  1          2          3     ...    20
price:  [lag_20,   lag_19,   lag_18,  ...,  lag_1]
          ↑ oldest                          ↑ most recent
```

The GRU processes this sequence step by step, updating its internal hidden state at each timestep. This lets it learn **temporal patterns** — e.g. "three consecutive rises followed by a plateau tends to precede a drop" — which the trees cannot detect because they see all lags simultaneously with no ordering.

### Why 100 lag columns but only 20 GRU timesteps?

- **100 lags (`SEQ_LEN`)** is the width of the feature table, giving the tree models a rich 100-day view of price history as flat inputs.
- **20 lookback (`LOOKBACK`)** is the window the GRU reads *as a sequence*. A 100-step GRU sequence would be 5× harder to train and more prone to vanishing gradients. 20 days (~1 trading month) captures enough short-to-medium-term momentum for the gating mechanism to be effective. The GRU simply ignores `lag_21` through `lag_100`.

### Sliding window

The ~4 100 training rows are created by a sliding window over the full history:

```
Row 1:    [day_1,  day_2,  ..., day_20]  → predict day_21
Row 2:    [day_2,  day_3,  ..., day_21]  → predict day_22
Row 3:    [day_3,  day_4,  ..., day_22]  → predict day_23
...
Row 4100: [day_4081, ..., day_4100]      → predict day_4101
```

Each day in the history appears in multiple windows, so the GRU effectively learns from every 20-day pattern that has ever occurred in BTC's price history.

At **inference time**, there is only one window — the most recent 20 days — and the GRU runs on that single row to produce tomorrow's prediction.

---

## Target Variable

```
y = log(close[t] / close[t-1])
```

Log-returns are stationary and approximately normally distributed, unlike raw prices which drift over time. This solves the tree-model extrapolation problem: predicting a small percentage change (bounded near zero) is well within the training distribution even when prices reach new all-time highs.

Predictions are back-transformed to prices using:

```
price[t] = close[t-1] × exp(predicted_log_return)
```

---

## Evaluation

### Train / Test Split
- **Method:** Chronological 70/30 split (no shuffling — avoids future data leakage)
- **Training set:** ~2015–2022
- **Test set:** ~2022–2026

### Metrics (computed in price domain after reconstruction)

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Average prediction error in USD; penalizes large errors more than MAE |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute prediction error in USD |
| **R²** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Proportion of variance explained; 1.0 is perfect, 0 means the model is no better than predicting the mean |

### Evaluation Protocols

**Holdout** — single pass over the test set. Fast, gives a baseline comparison.

**Walk-forward cross-validation** — simulates real deployment:
1. Start with training data only
2. Predict the next step
3. Add that step to the training window
4. Retrain tree models every `K=7` steps (GRU is not retrained per step due to cost)
5. Repeat for up to `max_iters=200` steps

Walk-forward is more realistic than holdout because the model is evaluated in sequence, never seeing future data at any step. The best model is selected based on walk-forward RMSE.

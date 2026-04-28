# Honest Model Evaluation — Why R²=0.99 Doesn't Mean What You Think

This document walks through a key insight that came out of reviewing this project: **the original evaluation metrics looked great, but they were measuring the wrong thing.** It explains the problem, the fix, and what the honest numbers actually say about the model.

---

## TL;DR

- The original holdout reported **R² = 0.9965** and **RMSE = $1,747** for the GRU.
- A do-nothing baseline (`predicted_price = previous_close`) scores **R² = 0.9967** and **RMSE = $1,733**.
- The "champion" GRU is therefore **0.8% worse than predicting no change at all**.
- This isn't a bug — it's the honest answer for daily Bitcoin prediction with only price-derived features.
- The fix is to add three things to every evaluation: a **persistence baseline**, **log-return-space metrics**, and **directional accuracy**.

---

## 1. The original picture

The notebook trained three models — Random Forest, XGBoost, GRU — on **log-returns** of the Bitcoin closing price. Predictions were back-transformed to dollars and scored:

```
=== Holdout (price-reconstructed from predicted log-returns) ===
            RMSE      MAE      R2
RF       2801.07  2057.78  0.9915
XGBoost  2165.73  1539.57  0.9949
GRU      1794.25  1214.71  0.9965
```

R² near 1.0 across the board, GRU as the headline winner. Looks excellent — but those numbers don't tell you what you think they tell you.

---

## 2. Why those metrics are misleading

The model is trained to predict a log-return:

```
y = log(close[t] / close[t-1])
```

…which we then back-transform to a price for evaluation:

```
predicted_price[t] = close[t-1] · exp(predicted_log_return[t])
```

Look closely at that formula. The predicted price is **yesterday's close times a tiny number near 1.0**. Daily Bitcoin log-returns have a standard deviation of about 0.038, so `exp(predicted_log_return)` is almost always between 0.96 and 1.04.

That means almost all of the variance in `predicted_price` comes from `close[t-1]` — a number we already have, that the model didn't have to learn. The model's actual contribution is the small `exp(...)` correction on top.

Now recall what R² measures:

```
R² = 1 - SS_res / SS_tot

SS_tot = Σ (actual_price - mean_actual_price)²
SS_res = Σ (actual_price - predicted_price)²
```

In price space:

- `SS_tot` is huge — Bitcoin ranged from $20k to $108k across the test window.
- `SS_res` is tiny — predictions hug yesterday's close, which is always close to today's actual price.

So `R²` is forced near 1.0 for **any** function that doesn't actively damage `close[t-1]`. Including a function that just returns `close[t-1]` directly.

In other words: the metric is grading the model on a question it didn't have to answer.

---

## 3. The fix — three things to add

To get an honest picture, we need to score the model in a way that strips away the "free lunch" of yesterday's close.

### 3a. Persistence baseline

Add a fourth "model" to the comparison that does nothing:

```python
predicted_log_return = 0   for every day
predicted_price      = close[t-1]
```

If a trained model can't beat this baseline on price-space RMSE, it isn't really learning anything beyond the fact that Bitcoin prices are autocorrelated (which we already knew).

### 3b. Log-return-space metrics

Score the model's **raw log-return prediction** directly against the actual log-return — without back-transforming to a price:

```python
r_true = y_test                                      # actual raw log-returns
r_pred = unscale_logret(model.predict(X_test_s))     # model output, just inverse-scaled
```

Then compute RMSE / MAE / R² on these. Now:

- **R² baseline = 0** — that means "as good as predicting the historical mean log-return" (≈ +0.0015).
- **R² < 0** is possible and meaningful — it means the model is *worse* than predicting that mean.
- A meaningful R² for crypto is typically in the 0.00–0.02 range. Anything above 0.05 should make you suspect data leakage.

This is the metric that measures whether the model has any *actual* skill on the prediction problem you set it.

### 3c. Directional accuracy

Strip magnitude entirely and just ask: **did you get up vs down right?**

```python
accuracy = mean( sign(predicted_log_return) == sign(actual_log_return) ) * 100
```

- Baseline = ~50% (coin flip). For BTC the slight upward drift makes ~52% the realistic baseline.
- 53–55% = small but real signal — often enough for a profitable trading strategy.
- > 57% with only price features = check for data leakage.

---

## 4. What changed in the code

All changes live in **`notebooks/01-compare-models.ipynb`, Cell 6 (Holdout evaluation)**. They are **purely additive** — no existing functionality was removed and downstream cells (walk-forward, plots, naive comparison, model selection) still work unchanged.

### 4a. Two new helper functions

```python
def metrics_logret(r_true, r_pred):
    """Same RMSE / MAE / R² formulas as metrics(), just on log-returns."""
    rmse = float(np.sqrt(np.mean((r_true - r_pred) ** 2)))
    mae  = float(mean_absolute_error(r_true, r_pred))
    r2   = float(r2_score(r_true, r_pred))
    return dict(RMSE=round(rmse, 5), MAE=round(mae, 5), R2=round(r2, 4))


def directional_accuracy(r_true, r_pred):
    """% of days the predicted sign matches the actual sign of the log-return."""
    return round(float(np.mean(np.sign(r_true) == np.sign(r_pred))) * 100, 2)
```

`metrics_logret` is mathematically identical to `metrics`. The only differences are higher decimal precision (log-return values are ~0.04, so 2 decimals would round everything to 0.04) and the parameter names signal the expected units.

### 4b. A new prediction-extraction helper

```python
def unscale_logret(scaled_pred):
    """Inverse-scale the model output → raw log-return (no price reconstruction)."""
    return scaler_y.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()
```

This is what makes the log-return-space evaluation possible. The original `inv_logret()` does inverse-scale **and then** reconstructs to a price (`close_prev · exp(...)`). `unscale_logret()` stops after the inverse-scale, giving us the model's prediction in its native units.

### 4c. Persistence baseline

```python
all_preds["Persistence"] = (np.zeros_like(r_true), close_prev_test.copy())
```

Predict log-return = 0 every day → predicted price = previous close. Free, deterministic, and the floor every trained model has to beat.

### 4d. Two output tables instead of one

The cell now prints:

1. **Price-space table** with RF, XGBoost, GRU, **and Persistence** — same metric the project always reported, but now with the baseline visible right next to the trained models.
2. **Log-return-space table** with RMSE / R² / **DirAcc%** for all four — the honest skill measure.
3. A **headline summary** comparing the GRU's price RMSE directly to persistence, with the % improvement (or regression).

---

## 5. The actual results

Running the upgraded Cell 6 on the current data:

### Price space (USD)

```
                RMSE      MAE      R2
RF           2925.74  2153.19  0.9907
XGBoost      2174.60  1547.92  0.9948
GRU          1747.34  1172.88  0.9967
Persistence  1733.47  1153.99  0.9967    ← do-nothing baseline
```

The persistence baseline ties the GRU on R² (both 0.9967) and **beats** it on RMSE by $13.87. The trees are actively *worse* than persistence — they introduce noise on top of yesterday's close.

### Log-return space

```
              RMSE      R2     DirAcc%
RF          0.03678  -1.1913   52.58
XGBoost     0.03059  -0.5162   49.67
GRU         0.02494  -0.0073   49.58
Persistence 0.02487  -0.0022    NaN
```

What this says:

- **All three trained models have negative R² in log-return space.** Each one is *worse than predicting the mean log-return*. They have anti-skill on magnitude.
- **GRU's RMSE (0.02494) is statistically identical to Persistence's (0.02487).** The GRU has converged to "predict ~0 every day," which is indistinguishable from doing nothing.
- **Directional accuracy** is essentially 50% for XGBoost and GRU — coin flips. RF is mildly above chance at 52.58% (see the next section for why this is interesting).

### Headline

```
GRU vs Persistence baseline (price-space RMSE):
  GRU         : $  1,747.34
  Persistence : $  1,733.47
  Improvement : $    -13.87  (-0.8%)
```

The "champion" model is 0.8% **worse** than doing nothing.

---

## 6. The one interesting nuance

| Model | Best at | Worst at |
|---|---|---|
| **RF** | Direction (52.58%) | Magnitude (R² = −1.19) |
| **GRU** | Magnitude (R² ≈ 0) | Direction (49.58%) |

Different metrics measure different things. RF makes large, confident predictions — usually wrong in size, but slightly better than chance at calling the direction. GRU collapsed to predicting near-zero every day — which gives small RMSE but a coin flip on direction.

**If the goal were "build a directional trading signal," RF would be the more interesting candidate** even though it "lost" the regression contest. This is a great talking point for the capstone presentation, and a real lesson in why you should always evaluate models with multiple complementary metrics.

---

## 7. Why this is good news for the capstone

This finding is *more* impressive than a fake 99% R², not less. Here's what you've actually demonstrated:

1. **Honest metrics that survive scrutiny.** Any reviewer with financial-ML experience would immediately ask "did you compare to a persistence baseline?" — and now you have a clean answer.
2. **A vocabulary** for talking about persistence baselines, log-return space vs price space, directional accuracy, and the difference between "explained variance" and "actual predictive skill."
3. **A clean narrative** for the presentation:
   > "I built three models, evaluated them properly with both price-space and log-return-space metrics, and found that none of them beat a persistence baseline on daily BTC. This is the expected result for daily price prediction with only price-derived features (consistent with the Efficient Market Hypothesis). Here's what would be needed to extract real signal."
4. **The MLOps pipeline is unaffected.** Docker, FastAPI, Oracle Cloud, MLflow tracking, drift detection, GitHub Actions retraining — every piece of the deployment work is still 100% valid. The deployment skill is the real point of this capstone, not model accuracy on a fundamentally hard problem.

Going from *"my model has 99.65% R²"* to *"my model is statistically indistinguishable from a no-skill baseline, and here's exactly why that's the expected result for this problem"* is a major level-up in maturity. It's what separates a textbook submission from one that demonstrates real engineering judgment.

---

## 8. Concrete paths forward

Three options, in order of effort:

### Option 1 — Don't change the model. Change the story. ★ Recommended

For a capstone with a tight deadline, this is the right answer. Update:

- `README.md` — report the persistence baseline alongside every metric.
- `docs/model-overview.md` — add a section on baseline-aware evaluation (similar to this document).
- `dashboard/components/Dashboard.tsx` — add a "vs persistence baseline" KPI next to the Test RMSE.
- `src/training/retrain.py` — reframe the drift threshold as "retrain when the model falls behind the persistence baseline."

Minimum extra ML work, maximum honesty payoff. The project becomes "honest baseline-aware financial forecasting on cloud infrastructure" — a much more defensible positioning than "99% accurate Bitcoin predictor."

### Option 2 — Add features outside of price

Price-derived features alone are very limited. Things that *might* unlock real signal:

- **On-chain metrics** — active addresses, hash rate, exchange flows (Glassnode, IntoTheBlock APIs).
- **Macro indicators** — DXY, S&P 500, gold, real yields.
- **Sentiment** — Fear & Greed Index, Twitter/Reddit volume, Google Trends for "Bitcoin."

Even if results stay weak, the *experiment* of adding them is great capstone material.

### Option 3 — Change the prediction target

Daily next-day price is genuinely hard. Easier (and more useful) targets:

- **Volatility forecasting** — much more predictable than returns, and there's well-known prior art (GARCH, realized volatility).
- **Longer horizons** — 1-week or 1-month forecasts have more signal because short-term noise averages out.
- **Directional classification** — "will tomorrow's return be positive?" with F1 / hit-rate metrics. Plays to RF's strength.

---

## 9. Where to look in the code

| What you want to read | Location |
|---|---|
| The "why two spaces" explanation in writing | `notebooks/01-compare-models.ipynb` → markdown cell **"## Evaluation Strategy"** → "Why two spaces?" subsection |
| New helper functions | Same notebook → next code cell (the one starting `# ── Cell 6: Holdout evaluation ──`) → look at `metrics_logret`, `directional_accuracy`, `unscale_logret` |
| Persistence baseline injection | Same cell → line `all_preds["Persistence"] = (np.zeros_like(r_true), close_prev_test.copy())` |
| New printed tables | Same cell → bottom `print(holdout_full_df.to_string())` and `print(logret_df.to_string())` blocks |
| Headline GRU vs Persistence summary | Same cell → very last `print` block computing `delta` and `pct` |

---

## 10. Suggested next steps (if you want to extend the same upgrade)

The same pattern — add persistence baseline, add log-return-space metrics, add directional accuracy — should also be applied to:

1. **Cell 7 (Walk-forward evaluation)** in the same notebook. Same logic, just inside the loop.
2. **The plots in Cell 9.** Add a 4th bar group ("Persistence") to the RMSE/R² charts so the floor is visible. Add a second figure for log-return-space metrics + directional accuracy with a 50% reference line.
3. **`notebooks/02-bitcoin-price-prediction.ipynb`**, Step 4 — this is the notebook that actually trains and saves `best_model.keras` and logs to MLflow, so the metrics it records should be the honest ones too.
4. **`README.md`** — the "Test RMSE ~$622" claim around lines 96–113 should be reframed alongside the persistence baseline so reviewers see the full picture.

But none of this is strictly necessary — Cell 6 alone is enough to teach the lesson and update the project's narrative.

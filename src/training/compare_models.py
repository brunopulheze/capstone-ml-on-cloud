"""Train and compare RandomForest, XGBoost, and GRU on engineered BTC features.

Features:
- Lagged window of past `SEQ_LEN` closes (flattened)
- RSI(14), MACD(12,26,9), rolling std(30), daily returns

Evaluation:
- Hold-out test (train on 70%, test on 30%) -- log-return space + persistence baseline
- Walk-forward CV (RF and XGB retrained each step; GRU trained once)

Outputs:
- Prints log-return metrics (RMSE, R2, DirAcc%) with persistence baseline
- Writes models/selection.json with the best model by log-return RMSE
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import xgboost as xgb
except Exception:
    xgb = None

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping


SEQ_LEN  = 100   # lag window for tree models
LOOKBACK = 20    # GRU sequence length


# ── Data / features ───────────────────────────────────────────────────────────

def download_close(ticker="BTC-USD", start="2015-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df[["Date", "Close"]]


def rsi(series, period=14):
    delta = series.diff()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    return 100 - (100 / (1 + up / down))


def macd(series, a=12, b=26, c=9):
    ema_a = series.ewm(span=a, adjust=False).mean()
    ema_b = series.ewm(span=b, adjust=False).mean()
    line  = ema_a - ema_b
    sig   = line.ewm(span=c, adjust=False).mean()
    return line, sig


def engineer_features(df, seq_len=SEQ_LEN):
    d = df.copy()
    d["return"]   = d["Close"].pct_change().shift(1)
    d["std30"]    = d["Close"].rolling(30).std().shift(1)
    d["rsi14"]    = rsi(d["Close"]).shift(1)
    ml, ms        = macd(d["Close"])
    d["macd"]     = ml.shift(1)
    d["macd_sig"] = ms.shift(1)
    d = d.dropna().reset_index(drop=True)
    for i in range(1, seq_len + 1):
        d[f"lag_{i}"] = d["Close"].shift(i)
    d = d.dropna().reset_index(drop=True)
    return d


def feature_cols(seq_len=SEQ_LEN):
    lag_cols   = [f"lag_{i}" for i in range(1, seq_len + 1)]
    extra_cols = ["std30", "rsi14", "macd", "macd_sig", "return"]
    return lag_cols + extra_cols


def gru_seq(X_scaled, lookback=LOOKBACK):
    """Reshape scaled features to (n, lookback, 1) for GRU input."""
    idx = list(range(lookback - 1, -1, -1))  # oldest lag first
    return X_scaled[:, idx].reshape(-1, lookback, 1)


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics_logret(r_true, r_pred):
    """Log-return space metrics -- the honest skill measure."""
    rmse = float(np.sqrt(np.mean((r_true - r_pred) ** 2)))
    mae  = float(mean_absolute_error(r_true, r_pred))
    r2   = float(r2_score(r_true, r_pred))
    return {"RMSE": round(rmse, 5), "MAE": round(mae, 5), "R2": round(r2, 4)}


def dir_acc(r_true, r_pred):
    return round(float(np.mean(np.sign(r_true) == np.sign(r_pred))) * 100, 2)


# ── Holdout ────────────────────────────────────────────────────────────────────

def train_holdout(df, seq_len=SEQ_LEN):
    fcols = feature_cols(seq_len)
    X_raw = df[fcols].values
    y_raw = np.log(df["Close"].values / df["lag_1"].values)  # log-return target

    split = int(len(X_raw) * 0.7)
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    scaler_X = MinMaxScaler()
    X_tr_s   = scaler_X.fit_transform(X_train)
    X_te_s   = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_tr_s   = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr_s)
    rf_r_pred = scaler_y.inverse_transform(rf.predict(X_te_s).reshape(-1, 1)).ravel()

    # XGBoost
    xg = None
    if xgb is not None:
        xg = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=4, verbosity=0)
        xg.fit(X_tr_s, y_tr_s)
        xg_r_pred = scaler_y.inverse_transform(xg.predict(X_te_s).reshape(-1, 1)).ravel()
    else:
        xg_r_pred = None

    # GRU
    X_tr_gru = gru_seq(X_tr_s, LOOKBACK)
    X_te_gru = gru_seq(X_te_s, LOOKBACK)
    gru_model = Sequential([GRU(64, input_shape=(LOOKBACK, 1)), Dense(1)])
    gru_model.compile(optimizer="adam", loss="mse")
    gru_model.fit(
        X_tr_gru, y_tr_s,
        epochs=50, batch_size=32, validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    gru_r_pred = scaler_y.inverse_transform(
        gru_model.predict(X_te_gru, verbose=0).ravel().reshape(-1, 1)
    ).ravel()

    # Persistence baseline: predict log-return = 0
    pers_r_pred = np.zeros_like(y_test)

    preds = {"RF": rf_r_pred, "XGBoost": xg_r_pred, "GRU": gru_r_pred, "Persistence": pers_r_pred}
    results = {}
    for name, r_pred in preds.items():
        if r_pred is None:
            continue
        row = metrics_logret(y_test, r_pred)
        row["DirAcc%"] = dir_acc(y_test, r_pred) if name != "Persistence" else float("nan")
        results[name] = row

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, os.path.join("models", "rf_model.save"))
    if xg is not None:
        joblib.dump(xg, os.path.join("models", "xgb_model.save"))
    gru_model.save(os.path.join("models", "keras_model.h5"))
    joblib.dump(scaler_X, os.path.join("models", "scaler.save"))

    return results, scaler_X, scaler_y, gru_model


# ── Walk-forward ───────────────────────────────────────────────────────────────

def walk_forward_models(df, scaler_X_trained, scaler_y_trained, gru_model_trained,
                        seq_len=SEQ_LEN, max_iters=200):
    """Walk-forward CV: tree models retrain each step; GRU is fixed."""
    fcols = feature_cols(seq_len)
    X_raw = df[fcols].values
    y_raw = np.log(df["Close"].values / df["lag_1"].values)

    split = int(len(X_raw) * 0.7)
    n     = min(len(X_raw) - split, max_iters) if max_iters else len(X_raw) - split

    X_tr_s = scaler_X_trained.transform(X_raw[:split])
    X_te_s = scaler_X_trained.transform(X_raw[split:split + n])
    y_tr_s = scaler_y_trained.transform(y_raw[:split].reshape(-1, 1)).ravel()
    y_te   = y_raw[split:split + n]

    preds_r = {"RF": [], "XGBoost": [], "GRU": [], "Persistence": []}
    Xw, yw  = X_tr_s.copy(), y_tr_s.copy()

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xw, yw)
    xg = None
    if xgb is not None:
        xg = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=4, verbosity=0)
        xg.fit(Xw, yw)

    for i in range(n):
        xi = X_te_s[i:i + 1]
        preds_r["RF"].append(
            scaler_y_trained.inverse_transform(rf.predict(xi).reshape(-1, 1)).ravel()[0]
        )
        if xg is not None:
            preds_r["XGBoost"].append(
                scaler_y_trained.inverse_transform(xg.predict(xi).reshape(-1, 1)).ravel()[0]
            )
        preds_r["GRU"].append(
            scaler_y_trained.inverse_transform(
                gru_model_trained.predict(gru_seq(xi, LOOKBACK), verbose=0).ravel().reshape(-1, 1)
            ).ravel()[0]
        )
        preds_r["Persistence"].append(0.0)

        Xw = np.vstack([Xw, xi])
        yw = np.append(yw, scaler_y_trained.transform([[y_te[i]]]).ravel()[0])
        rf.fit(Xw, yw)
        if xg is not None:
            xg.fit(Xw, yw)

        if (i + 1) % 50 == 0:
            print(f"  walk-forward step {i + 1}/{n}")

    results = {}
    for name, r_preds in preds_r.items():
        if not r_preds:
            continue
        r_arr = np.array(r_preds)
        row = metrics_logret(y_te, r_arr)
        row["DirAcc%"] = dir_acc(y_te, r_arr) if name != "Persistence" else float("nan")
        results[name] = row
    return results


# ── Model selection ────────────────────────────────────────────────────────────

def choose_best_and_write(hold_results, wf_results):
    """Select the model with the lowest log-return RMSE (walk-forward preferred)."""
    src = wf_results if wf_results else hold_results
    candidates = {n: v for n, v in src.items() if n != "Persistence" and v is not None}
    if not candidates:
        print("No valid candidates -- skipping selection.")
        return None, {}
    best   = min(candidates, key=lambda n: candidates[n]["RMSE"])
    scores = {n: v["RMSE"] for n, v in candidates.items()}
    selection = {"model": best, "logret_rmse": candidates[best]["RMSE"]}
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "selection.json"), "w") as f:
        json.dump(selection, f, indent=2)
    return best, scores


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Downloading data...")
    df_raw  = download_close()
    df_feat = engineer_features(df_raw)

    print("Running holdout training (log-return target)...")
    hold_res, scaler_X, scaler_y, gru_m = train_holdout(df_feat)

    print("\nResults -- Holdout (log-return space):")
    for k, v in hold_res.items():
        if v is None:
            print(f"  {k}: skipped")
        else:
            nan = float("nan")
            da = f"  DirAcc={v['DirAcc%']:.1f}%" if v['DirAcc%'] == v['DirAcc%'] else ""
            print(f"  {k}: RMSE={v['RMSE']:.5f}  R2={v['R2']:.4f}{da}")

    print("\nRunning walk-forward CV (200 steps)...")
    wf_res = walk_forward_models(df_feat, scaler_X, scaler_y, gru_m, max_iters=200)

    print("\nResults -- Walk-forward (log-return space):")
    for k, v in wf_res.items():
        if v is None:
            print(f"  {k}: skipped")
        else:
            da = f"  DirAcc={v['DirAcc%']:.1f}%" if v['DirAcc%'] == v['DirAcc%'] else ""
            print(f"  {k}: RMSE={v['RMSE']:.5f}  R2={v['R2']:.4f}{da}")

    best, scores = choose_best_and_write(hold_res, wf_res)
    print(f"\nSelected best model: {best}  (log-return RMSE: {scores})")


if __name__ == "__main__":
    main()

"""Train and compare RandomForest, XGBoost, and LSTM on engineered BTC features.

Features:
- Lagged window of past `SEQ_LEN` closes (flattened)  
- 100-day rolling average (shifted)  
- RSI(14), MACD(12,26,9), rolling std(30), daily returns

Evaluations:
- Hold-out test (train on 70%, test on 30%)
- Walk-forward CV (RF & XGB retrained each step; LSTM trained once)

Outputs:
- Prints metrics and writes `models/selection.json` with the chosen model.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
except Exception:
    xgb = None

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


SEQ_LEN = 100


def download_close(ticker: str = "BTC-USD", start: str = "2015-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start)
    df = df.reset_index()
    return df[["Date", "Close"]]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, a=12, b=26, c=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_a = series.ewm(span=a, adjust=False).mean()
    ema_b = series.ewm(span=b, adjust=False).mean()
    macd_line = ema_a - ema_b
    signal = macd_line.ewm(span=c, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["return"] = d["Close"].pct_change()
    d["ma100_prev"] = d["Close"].rolling(100).mean().shift(1)
    d["std30_prev"] = d["Close"].rolling(30).std().shift(1)
    d["rsi14"] = rsi(d["Close"]).shift(1)
    macd_line, signal, hist = macd(d["Close"]) 
    d["macd"] = macd_line.shift(1)
    d["macd_sig"] = signal.shift(1)
    d = d.dropna().reset_index(drop=True)
    return d


def make_lag_features(d: pd.DataFrame, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    # create lagged close columns: lag_1 .. lag_SEQ_LEN (most recent last)
    df = d.copy()
    for i in range(1, seq_len + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df


def prepare_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_feat = engineer_features(df)
    df_lag = make_lag_features(df_feat)

    feature_cols = [f"lag_{i}" for i in range(1, SEQ_LEN + 1)] + ["ma100_prev", "std30_prev", "rsi14", "macd", "macd_sig", "return"]
    X = df_lag[feature_cols].copy()
    y = df_lag["Close"].copy()
    return X, y, df_lag, feature_cols


def train_holdout(X: pd.DataFrame, y: pd.Series):
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # XGBoost
    if xgb is not None:
        xg = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=4, verbosity=0)
        xg.fit(X_train, y_train)
        xg_pred = xg.predict(X_test)
    else:
        xg = None
        xg_pred = None

    # LSTM: prepare scaled sequences
    scaler = MinMaxScaler()
    X_vals = scaler.fit_transform(X_train)
    # build sequences for LSTM using only Close lags (reorder columns where lag_1..lag_N are first)
    lag_cols = [f"lag_{i}" for i in range(1, SEQ_LEN + 1)]
    X_train_seq = X_train[lag_cols].values.reshape((X_train.shape[0], SEQ_LEN, 1))
    X_test_seq = X_test[lag_cols].values.reshape((X_test.shape[0], SEQ_LEN, 1))

    model = Sequential([LSTM(64, input_shape=(SEQ_LEN, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_seq, y_train.values, epochs=3, batch_size=64, verbose=0)
    lstm_pred = model.predict(X_test_seq).ravel()

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2}

    results = {
        "rf": metrics(y_test.values, rf_pred),
        "xgb": metrics(y_test.values, xg_pred) if xg is not None else None,
        "lstm": metrics(y_test.values, lstm_pred),
    }

    # Save trained models and scaler
    os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)
    import joblib

    joblib.dump(rf, os.path.join("models", "rf_model.save"))
    if xg is not None:
        joblib.dump(xg, os.path.join("models", "xgb_model.save"))
    model.save(os.path.join("models", "keras_model.h5"))
    joblib.dump(scaler, os.path.join("models", "scaler.save"))

    return results


def walk_forward_models(X: pd.DataFrame, y: pd.Series, retrain_steps=(1, 7, 30), max_iters: int | None = None):
    # We'll do walk-forward predictions retraining RF and XGB at each step; LSTM trained once on initial train
    split = int(len(X) * 0.7)
    start = split
    preds = {"rf": [], "xgb": [], "lstm": []}
    trues = []

    # train initial LSTM on train portion
    lag_cols = [f"lag_{i}" for i in range(1, SEQ_LEN + 1)]
    X_train0 = X.iloc[:split]
    y_train0 = y.iloc[:split]
    X_train_seq = X_train0[lag_cols].values.reshape((X_train0.shape[0], SEQ_LEN, 1))
    lstm = Sequential([LSTM(64, input_shape=(SEQ_LEN, 1)), Dense(1)])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train_seq, y_train0.values, epochs=3, batch_size=64, verbose=0)

    iters = 0
    for t in range(split, len(X)):
        X_train = X.iloc[:t]
        y_train = y.iloc[:t]
        X_test_row = X.iloc[t:t+1]
        y_true = y.iloc[t]

        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test_row)[0]

        if xgb is not None:
            xg = xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=4, verbosity=0)
            xg.fit(X_train, y_train)
            xg_pred = xg.predict(X_test_row)[0]
        else:
            xg_pred = None

        # LSTM predict using last seq from combined data up to t
        last_seq = X.iloc[t][lag_cols].values.reshape(1, SEQ_LEN, 1)
        lstm_pred = lstm.predict(last_seq)[0, 0]

        preds["rf"].append(rf_pred)
        preds["xgb"].append(xg_pred)
        preds["lstm"].append(lstm_pred)
        trues.append(y_true)

        iters += 1
        if max_iters and iters >= max_iters:
            break

    def agg(p_list):
        if p_list is None or len(p_list) == 0:
            return None
        mae = mean_absolute_error(trues, p_list)
        rmse = float(np.sqrt(np.mean((np.array(trues) - np.array(p_list)) ** 2)))
        r2 = r2_score(trues, p_list)
        return {"n_samples": len(trues), "mae": mae, "rmse": rmse, "r2": r2}

    return {"rf": agg(preds["rf"]), "xgb": agg(preds["xgb"]), "lstm": agg(preds["lstm"])}


def choose_best_and_write(results_holdout: dict, results_wf: dict):
    # prefer walk-forward r2 if available, fall back to holdout
    scores = {}
    for model in ("rf", "xgb", "lstm"):
        wf = results_wf.get(model)
        hold = results_holdout.get(model)
        if wf and wf.get("r2") is not None:
            scores[model] = wf["r2"]
        elif hold and hold.get("r2") is not None:
            scores[model] = hold["r2"]
        else:
            scores[model] = -999

    best = max(scores, key=scores.get)
    selection = {"model": best}
    with open(os.path.join("models", "selection.json"), "w") as f:
        json.dump(selection, f)
    return best, scores


def main():
    print("Downloading data and preparing features...")
    df = download_close()
    X, y, df_lag, feature_cols = prepare_datasets(df)

    print("Running holdout training...")
    hold_res = train_holdout(X, y)

    print("Running walk-forward CV (this will retrain RF and XGB at each step; may take time) ...")
    wf_res = walk_forward_models(X, y, max_iters=None)

    print("Results — Holdout:")
    for k, v in hold_res.items():
        if v is None:
            print(f"- {k}: skipped")
        else:
            print(f"- {k}: MAE=${v['mae']:.2f}, RMSE=${v['rmse']:.2f}, R²={v['r2']:.4f}")

    print("\nResults — Walk-forward:")
    for k, v in wf_res.items():
        if v is None:
            print(f"- {k}: skipped")
        else:
            print(f"- {k}: samples={v['n_samples']}, MAE=${v['mae']:.2f}, RMSE=${v['rmse']:.2f}, R²={v['r2']:.4f}")

    best, scores = choose_best_and_write(hold_res, wf_res)
    print(f"\nSelected best model: {best} (scores: {scores})")


if __name__ == '__main__':
    main()

"""Compare Linear Regression feature choices:

- overlapping lagged windows (as in notebook)
- non-overlapping evaluation (step=SEQ_LEN on test)
- 100-day rolling average feature

Prints MAE, RMSE, R2 for each variant.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def create_sequences(values: np.ndarray, seq_len: int = 100, step: int = 1):
    X, y = [], []
    for i in range(seq_len, len(values), step):
        X.append(values[i - seq_len:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)


def download_close(ticker: str = "BTC-USD", start: str = "2015-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start)
    if df.empty:
        raise RuntimeError("Failed to download data; check Internet or yfinance")
    df = df.reset_index()
    return df[["Date", "Close"]]


def evaluate_lr_overlapping_vs_nonoverlapping(df: pd.DataFrame, seq_len: int = 100):
    data = df[["Close"]].copy()
    split_idx = int(len(data) * 0.7)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals = scaler.fit_transform(train)
    test_vals = scaler.transform(test)

    # prepare overlapping (step=1) sequences for train, and two test variants
    x_train, y_train = create_sequences(train_vals, seq_len=seq_len, step=1)

    # overlapping test sequences (as originally used)
    x_test_over, y_test_over = create_sequences(
        np.concatenate([train_vals[-seq_len:], test_vals], axis=0), seq_len=seq_len, step=1
    )

    # non-overlapping test sequences: step=seq_len to avoid overlap
    x_test_non, y_test_non = create_sequences(
        np.concatenate([train_vals[-seq_len:], test_vals], axis=0), seq_len=seq_len, step=seq_len
    )

    # Train LR on overlapping train sequences
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    def metrics(y_true_scaled, y_pred_scaled):
        y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    lr_pred_over = lr.predict(x_test_over)
    lr_pred_non = lr.predict(x_test_non)

    m_over = metrics(y_test_over, lr_pred_over)
    m_non = metrics(y_test_non, lr_pred_non)

    return {
        "overlapping": {"n_samples": len(y_test_over), "mae": m_over[0], "rmse": m_over[1], "r2": m_over[2]},
        "non_overlapping": {"n_samples": len(y_test_non), "mae": m_non[0], "rmse": m_non[1], "r2": m_non[2]},
    }


def evaluate_lr_rolling_avg(df: pd.DataFrame, window: int = 100):
    d = df.copy()
    # rolling mean up to previous day to avoid leakage
    d["ma_prev"] = d["Close"].rolling(window).mean().shift(1)
    d = d.dropna().reset_index(drop=True)

    split_idx = int(len(d) * 0.7)
    train = d.iloc[:split_idx]
    test = d.iloc[split_idx:]

    # scalers for X and y (y is Close)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[["ma_prev"]].values)
    y_train = scaler_y.fit_transform(train[["Close"]].values).ravel()

    X_test = scaler_X.transform(test[["ma_prev"]].values)
    y_test = scaler_y.transform(test[["Close"]].values).ravel()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_scaled = lr.predict(X_test)

    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)

    return {"n_samples": len(y_test), "mae": mae, "rmse": rmse, "r2": r2}


def evaluate_nonoverlapping_steps(df: pd.DataFrame, seq_len: int = 100, steps=(1, 7, 30, 100)):
    """Train on overlapping train sequences and evaluate on test with different step sizes."""
    data = df[["Close"]].copy()
    split_idx = int(len(data) * 0.7)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals = scaler.fit_transform(train)
    test_vals = scaler.transform(test)

    x_train, y_train = create_sequences(train_vals, seq_len=seq_len, step=1)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    results = {}
    concat = np.concatenate([train_vals[-seq_len:], test_vals], axis=0)
    for step in steps:
        x_test_step, y_test_step = create_sequences(concat, seq_len=seq_len, step=step)
        if len(y_test_step) == 0:
            results[step] = None
            continue
        pred = lr.predict(x_test_step)
        y_true = scaler.inverse_transform(y_test_step.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = r2_score(y_true, y_pred)
        results[step] = {"n_samples": len(y_test_step), "mae": mae, "rmse": rmse, "r2": r2}

    return results


def walk_forward_cv(df: pd.DataFrame, seq_len: int = 100, initial_train_frac: float = 0.7, max_iters: int | None = 500):
    """Simple walk-forward CV: expanding training window, predict next day using last seq_len values.

    This trains a fresh LR at each step (reasonable for LinearRegression) and avoids leakage by
    fitting scalers on the training window only.
    """
    closes = df[["Close"]].copy().reset_index(drop=True)
    n = len(closes)
    split_idx = int(n * initial_train_frac)
    start_t = split_idx + seq_len
    preds = []
    trues = []
    iters = 0
    for t in range(start_t, n):
        train = closes.iloc[:t]
        if len(train) < seq_len + 1:
            continue
        scaler = MinMaxScaler()
        train_vals = scaler.fit_transform(train)
        x_train, y_train = create_sequences(train_vals, seq_len=seq_len, step=1)
        if len(x_train) == 0:
            continue
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        last_seq = closes.iloc[t - seq_len:t]["Close"].values.reshape(-1, 1)
        last_seq_scaled = scaler.transform(last_seq).reshape(1, -1)
        pred_scaled = lr.predict(last_seq_scaled)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        true = closes.iloc[t]["Close"]
        preds.append(pred)
        trues.append(true)

        iters += 1
        if max_iters and iters >= max_iters:
            break

    if len(trues) == 0:
        return None

    mae = mean_absolute_error(trues, preds)
    rmse = float(np.sqrt(np.mean((np.array(trues) - np.array(preds)) ** 2)))
    r2 = r2_score(trues, preds)
    return {"n_samples": len(trues), "mae": mae, "rmse": rmse, "r2": r2}


def main():
    print("Downloading data and running comparisons — this may take a few seconds...")
    df = download_close()
    seq_len = 100

    res_seq = evaluate_lr_overlapping_vs_nonoverlapping(df, seq_len=seq_len)
    res_ma = evaluate_lr_rolling_avg(df, window=seq_len)
    res_steps = evaluate_nonoverlapping_steps(df, seq_len=seq_len, steps=(1, 7, 30, 100))
    res_wf = walk_forward_cv(df, seq_len=seq_len, initial_train_frac=0.7, max_iters=None)

    print("\nLinear Regression — overlapping vs non-overlapping (test) — seq_len=100")
    for k, v in res_seq.items():
        print(f"- {k}: samples={v['n_samples']}, MAE=${v['mae']:.2f}, RMSE=${v['rmse']:.2f}, R²={v['r2']:.4f}")

    print("\nLinear Regression — 100-day rolling-average feature")
    print(f"- rolling_avg: samples={res_ma['n_samples']}, MAE=${res_ma['mae']:.2f}, RMSE=${res_ma['rmse']:.2f}, R²={res_ma['r2']:.4f}")
    print("\nLinear Regression — non-overlapping with multiple steps")
    for step, val in res_steps.items():
        if val is None:
            print(f"- step={step}: no samples")
        else:
            print(f"- step={step}: samples={val['n_samples']}, MAE=${val['mae']:.2f}, RMSE=${val['rmse']:.2f}, R²={val['r2']:.4f}")

    print("\nLinear Regression — walk-forward CV (expanding training window)")
    if res_wf is None:
        print("- walk-forward: no samples / insufficient data")
    else:
        print(f"- walk-forward: samples={res_wf['n_samples']}, MAE=${res_wf['mae']:.2f}, RMSE=${res_wf['rmse']:.2f}, R²={res_wf['r2']:.4f}")


if __name__ == "__main__":
    main()

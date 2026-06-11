import numpy as np
import pandas as pd


def compute_daily_returns(df):
    return df["Close"].pct_change()


def compute_log_returns(df):
    return np.log(df["Close"] / df["Close"].shift(1))


def compute_moving_averages(df):
    out = pd.DataFrame(index=df.index)
    for window in [20, 50, 200]:
        out[f"MA{window}"] = df["Close"].rolling(window).mean()
    return out


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(df):
    ema12 = compute_ema(df["Close"], 12)
    ema26 = compute_ema(df["Close"], 26)
    macd = ema12 - ema26
    signal = compute_ema(macd, 9)
    return pd.DataFrame({"MACD": macd, "MACD_Signal": signal, "MACD_Hist": macd - signal}, index=df.index)


def compute_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(df, window=20, num_std=2):
    ma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    return pd.DataFrame({
        "BB_Upper": ma + num_std * std,
        "BB_Middle": ma,
        "BB_Lower": ma - num_std * std,
    }, index=df.index)


def compute_rolling_volatility(df, window=30):
    log_ret = compute_log_returns(df)
    return log_ret.rolling(window).std()


def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_lag_features(df, column="Close", lags=None):
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    out = pd.DataFrame(index=df.index)
    for lag in lags:
        out[f"{column}_lag{lag}"] = df[column].shift(lag)
    return out


def engineer_all_features(df):
    df = df.sort_values("Date").copy()
    df["Daily_Return"] = compute_daily_returns(df)
    df["Log_Return"] = compute_log_returns(df)

    mas = compute_moving_averages(df)
    for col in mas.columns:
        df[col] = mas[col]

    macd = compute_macd(df)
    for col in macd.columns:
        df[col] = macd[col]

    df["RSI"] = compute_rsi(df)

    bb = compute_bollinger_bands(df)
    for col in bb.columns:
        df[col] = bb[col]

    df["Rolling_Vol_30"] = compute_rolling_volatility(df)
    df["ATR_14"] = compute_atr(df)

    lags = add_lag_features(df)
    for col in lags.columns:
        df[col] = lags[col]

    return df

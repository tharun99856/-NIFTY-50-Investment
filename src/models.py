import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLS_BASELINE = ["Close_lag1", "Close_lag2", "Close_lag3", "MA20", "Volume"]

FEATURE_COLS_XGBOOST = [
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5", "Close_lag10",
    "MA20", "MA50", "Volume", "RSI", "MACD", "MACD_Hist",
    "BB_Upper", "BB_Lower", "Rolling_Vol_30", "ATR_14", "Daily_Return",
]


def time_split(df, test_frac=0.2):
    n = len(df)
    split = int(n * (1 - test_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def evaluate_predictions(y_true, y_pred):
    direction_true = np.sign(np.diff(np.concatenate([[y_true.iloc[0]], y_true.values])))
    direction_pred = np.sign(np.diff(np.concatenate([[y_true.iloc[0]], y_pred])))
    dir_acc = np.mean(direction_true == direction_pred)

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Directional_Accuracy": dir_acc,
    }


def train_baseline(train_df, feature_cols=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS_BASELINE
    cols = [c for c in feature_cols if c in train_df.columns]
    X = train_df[cols].dropna()
    y = train_df.loc[X.index, "Close"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_baseline(model, df, feature_cols=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS_BASELINE
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].dropna()
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="Predicted_Close")


def train_xgboost(train_df, feature_cols=None, regime_col=None, **kwargs):
    if feature_cols is None:
        feature_cols = FEATURE_COLS_XGBOOST
    cols = [c for c in feature_cols if c in train_df.columns]
    if regime_col and regime_col in train_df.columns:
        cols = cols + [regime_col]
    X = train_df[cols].dropna()
    y = train_df.loc[X.index, "Close"]

    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def predict_xgboost(model, df, feature_cols=None, regime_col=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS_XGBOOST
    cols = [c for c in feature_cols if c in df.columns]
    if regime_col and regime_col in df.columns:
        cols = cols + [regime_col]
    X = df[cols].dropna()
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="Predicted_Close")


def run_prediction_pipeline(stock_df, symbol, regime_col=None):
    train, test = time_split(stock_df)

    baseline = train_baseline(train)
    base_preds = predict_baseline(baseline, test)
    valid_idx = base_preds.index
    base_metrics = evaluate_predictions(test.loc[valid_idx, "Close"], base_preds)

    xgb = train_xgboost(train, regime_col=regime_col)
    xgb_preds = predict_xgboost(xgb, test, regime_col=regime_col)
    valid_idx_xgb = xgb_preds.index
    xgb_metrics = evaluate_predictions(test.loc[valid_idx_xgb, "Close"], xgb_preds)

    return {
        "symbol": symbol,
        "baseline_model": baseline,
        "xgboost_model": xgb,
        "baseline_metrics": base_metrics,
        "xgboost_metrics": xgb_metrics,
        "test_actual": test["Close"],
        "baseline_preds": base_preds,
        "xgboost_preds": xgb_preds,
        "train_size": len(train),
        "test_size": len(test),
    }

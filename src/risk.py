import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.06
TRADING_DAYS = 252


def annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns, risk_free=RISK_FREE_RATE):
    annual_ret = daily_returns.mean() * TRADING_DAYS
    annual_vol = annualized_volatility(daily_returns)
    if annual_vol == 0:
        return 0.0
    return (annual_ret - risk_free) / annual_vol


def sortino_ratio(daily_returns, risk_free=RISK_FREE_RATE):
    annual_ret = daily_returns.mean() * TRADING_DAYS
    neg = daily_returns[daily_returns < 0]
    if len(neg) == 0:
        return np.nan
    downside_dev = neg.std() * np.sqrt(TRADING_DAYS)
    if downside_dev == 0:
        return np.nan
    return (annual_ret - risk_free) / downside_dev


def maximum_drawdown(daily_returns):
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def value_at_risk(daily_returns, confidence=0.95):
    return daily_returns.quantile(1 - confidence)


def beta_vs_index(stock_returns, index_returns):
    aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if cov[1, 1] == 0:
        return np.nan
    return cov[0, 1] / cov[1, 1]


def compute_stock_risk_table(returns, index_returns=None):
    records = []
    for col in returns.columns:
        r = returns[col].dropna()
        record = {
            "Symbol": col,
            "Annual_Return": r.mean() * TRADING_DAYS,
            "Annual_Volatility": annualized_volatility(r),
            "Sharpe_Ratio": sharpe_ratio(r),
            "Sortino_Ratio": sortino_ratio(r),
            "Max_Drawdown": maximum_drawdown(r),
            "VaR_95": value_at_risk(r),
        }
        if index_returns is not None:
            record["Beta"] = beta_vs_index(r, index_returns)
        records.append(record)

    return pd.DataFrame(records).set_index("Symbol").round(4)


def compute_portfolio_risk(weights, returns):
    port_returns = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    return {
        "Annual_Return": round(port_returns.mean() * TRADING_DAYS, 4),
        "Annual_Volatility": round(annualized_volatility(port_returns), 4),
        "Sharpe_Ratio": round(sharpe_ratio(port_returns), 4),
        "Sortino_Ratio": round(sortino_ratio(port_returns), 4),
        "Max_Drawdown": round(maximum_drawdown(port_returns), 4),
        "VaR_95": round(value_at_risk(port_returns), 4),
    }


def detect_anomalies(daily_returns, threshold_std=3.0):
    mean = daily_returns.mean()
    std = daily_returns.std()
    return daily_returns[abs(daily_returns - mean) > threshold_std * std]


def regime_risk_comparison(returns, regimes, index_returns=None):
    results = {}
    for regime in regimes.unique():
        mask = regimes == regime
        regime_returns = returns.loc[mask]
        idx_ret = index_returns.loc[mask] if index_returns is not None else None
        if len(regime_returns) > 30:
            results[regime] = compute_stock_risk_table(regime_returns, idx_ret)
    return results

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RISK_FREE_RATE = 0.06
TRADING_DAYS = 252

SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "NTPC": "Energy", "POWERGRID": "Energy", "COALINDIA": "Energy", "GAIL": "Energy",
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "KOTAKBANK": "Financials",
    "SBIN": "Financials", "AXISBANK": "Financials", "INDUSINDBK": "Financials",
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "HDFCLIFE": "Financials",
    "SBILIFE": "Financials", "HDFC": "Financials",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "ADANIPORTS": "Infrastructure", "ULTRACEMCO": "Infrastructure",
    "GRASIM": "Infrastructure", "SHREECEM": "Infrastructure",
    "LT": "Infrastructure",
    "ASIANPAINT": "Consumer", "TITAN": "Consumer",
    "BHARTIARTL": "Telecom", "UPL": "Chemicals",
}


def get_sector(symbol):
    return SECTOR_MAP.get(symbol, "Other")


def compute_annual_stats(returns):
    stats = pd.DataFrame({
        "Annual_Return": returns.mean() * TRADING_DAYS,
        "Annual_Vol": returns.std() * np.sqrt(TRADING_DAYS),
    })
    stats["Sharpe"] = (stats["Annual_Return"] - RISK_FREE_RATE) / stats["Annual_Vol"].replace(0, np.nan)
    stats = stats.dropna()
    return stats


def inverse_volatility_weights(vol):
    inv_vol = 1.0 / vol
    return inv_vol / inv_vol.sum()


def momentum_weights(momentum):
    if len(momentum) == 0:
        return momentum
    pos_mom = momentum.clip(lower=0)
    if pos_mom.sum() == 0:
        return pd.Series(1.0 / len(momentum), index=momentum.index)
    return pos_mom / pos_mom.sum()


def apply_sector_constraint(weights, max_sector_pct=0.30):
    sectors = weights.index.map(get_sector)
    sector_weights = weights.groupby(sectors).transform("sum")
    for sector in sectors.unique():
        mask = sectors == sector
        total = weights[mask].sum()
        if total > max_sector_pct:
            weights[mask] *= max_sector_pct / total
    weights /= weights.sum()
    return weights


def apply_max_weight_constraint(weights, max_weight):
    capped = weights.clip(upper=max_weight)
    capped /= capped.sum()
    return capped


def build_conservative_portfolio(returns, lookback_days=252):
    recent = returns.iloc[-lookback_days:]
    stats = compute_annual_stats(recent)

    low_vol = stats.nsmallest(int(len(stats) * 0.25), "Annual_Vol")
    candidates = low_vol[low_vol["Sharpe"] > 1.0]
    if len(candidates) < 3:
        candidates = low_vol.nlargest(max(3, len(low_vol)), "Sharpe")

    weights = inverse_volatility_weights(candidates["Annual_Vol"])
    weights = apply_max_weight_constraint(weights, 0.10)
    weights = apply_sector_constraint(weights, 0.30)

    return {
        "name": "Conservative",
        "weights": weights,
        "stats": stats.loc[weights.index],
        "method": "Inverse-volatility weighting",
    }


def mean_variance_optimize(returns, target_return=None):
    returns = returns.dropna(axis=1, how="all")
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    n = len(mu)

    if n < 2:
        return pd.Series(1.0 / max(n, 1), index=mu.index)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(max(w @ cov @ w, 1e-10))
        return -(ret - RISK_FREE_RATE) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    bounds = [(0, 0.15)] * n
    x0 = np.ones(n) / n
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = pd.Series(result.x, index=mu.index)
    weights = weights[weights > 0.01]
    if weights.sum() == 0:
        weights = pd.Series(1.0 / n, index=mu.index)
    weights /= weights.sum()
    return weights


def build_balanced_portfolio(returns, lookback_days=252):
    recent = returns.iloc[-lookback_days:]
    stats = compute_annual_stats(recent)

    candidates = stats[(stats["Annual_Vol"] < stats["Annual_Vol"].quantile(0.75)) &
                       (stats["Sharpe"] > 0.5)]
    if len(candidates) < 5:
        candidates = stats.nlargest(10, "Sharpe")

    cand_returns = recent[candidates.index]
    weights = mean_variance_optimize(cand_returns)
    weights = apply_sector_constraint(weights, 0.30)

    return {
        "name": "Balanced",
        "weights": weights,
        "stats": stats.loc[weights.index],
        "method": "Mean-variance optimization (max Sharpe)",
    }


def build_aggressive_portfolio(returns, lookback_days=126):
    recent = returns.iloc[-lookback_days:]
    cum_returns = (1 + recent).prod() - 1
    cum_returns = cum_returns.dropna()
    stats = compute_annual_stats(returns.iloc[-252:])

    n_top = max(3, int(len(cum_returns) * 0.25))
    top_momentum = cum_returns.nlargest(n_top)
    weights = momentum_weights(top_momentum)
    weights = apply_max_weight_constraint(weights, 0.20)

    return {
        "name": "Aggressive",
        "weights": weights,
        "stats": stats.loc[weights.index],
        "method": "Momentum-weighted (6M return)",
    }


def portfolio_performance(weights, returns):
    port_returns = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    annual_ret = port_returns.mean() * TRADING_DAYS
    annual_vol = port_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol

    cumulative = (1 + port_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    neg_returns = port_returns[port_returns < 0]
    downside_std = neg_returns.std() * np.sqrt(TRADING_DAYS)
    sortino = (annual_ret - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan

    var_95 = port_returns.quantile(0.05)

    sectors = weights.index.map(get_sector)
    sector_breakdown = weights.groupby(sectors).sum()

    return {
        "Annual_Return": annual_ret,
        "Annual_Volatility": annual_vol,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Max_Drawdown": max_drawdown,
        "VaR_95": var_95,
        "Sector_Breakdown": sector_breakdown,
        "Top_5_Holdings": weights.nlargest(5),
        "Cumulative_Returns": cumulative,
        "Daily_Returns": port_returns,
    }


def build_regime_portfolios(returns, regimes):
    results = {}

    bull_mask = regimes == "Bull"
    bear_mask = regimes == "Bear"

    if bull_mask.sum() > 60:
        bull_returns = returns.loc[bull_mask]
        mom = (1 + bull_returns.iloc[-min(126, len(bull_returns)):]).prod() - 1
        top = mom.nlargest(int(len(mom) * 0.25))
        weights = momentum_weights(top)
        weights = apply_max_weight_constraint(weights, 0.20)
        results["Bull"] = {"weights": weights, "name": "Bull Regime Portfolio",
                           "method": "Momentum-weighted, higher concentration"}

    if bear_mask.sum() > 30:
        bear_returns = returns.loc[bear_mask]
        stats = compute_annual_stats(bear_returns)
        defensive_sectors = ["FMCG", "Pharma", "IT"]
        defensive = [s for s in stats.index if get_sector(s) in defensive_sectors]
        if len(defensive) >= 3:
            weights = inverse_volatility_weights(stats.loc[defensive, "Annual_Vol"])
        else:
            low_vol = stats.nsmallest(max(5, len(stats) // 4), "Annual_Vol")
            weights = inverse_volatility_weights(low_vol["Annual_Vol"])
        weights = apply_max_weight_constraint(weights, 0.15)
        results["Bear"] = {"weights": weights, "name": "Bear Regime Portfolio",
                           "method": "Defensive (FMCG/Pharma/IT), inverse-vol weighted"}

    sideways_mask = regimes == "Sideways"
    if sideways_mask.sum() > 60:
        sw_returns = returns.loc[sideways_mask]
        stats = compute_annual_stats(sw_returns)
        top_sharpe = stats.nlargest(10, "Sharpe")
        weights = mean_variance_optimize(sw_returns[top_sharpe.index])
        results["Sideways"] = {"weights": weights, "name": "Sideways Regime Portfolio",
                               "method": "Mean-variance optimized, sector-diversified"}

    return results

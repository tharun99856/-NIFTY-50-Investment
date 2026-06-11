import numpy as np
import pandas as pd
from .risk import sharpe_ratio, annualized_volatility, maximum_drawdown, value_at_risk


def generate_stock_recommendation(symbol, stock_df, regime, regime_confidence,
                                  all_regime_stats=None):
    recent = stock_df.tail(90)
    returns = recent["Daily_Return"].dropna()

    current_price = stock_df["Close"].iloc[-1]
    rsi = stock_df["RSI"].iloc[-1] if "RSI" in stock_df.columns else None
    sharpe_90d = sharpe_ratio(returns)
    vol_90d = annualized_volatility(returns)
    monthly_return = returns.tail(21).mean() * 21

    if regime == "Bull" and rsi and rsi < 70 and sharpe_90d > 0.5:
        action = "BUY"
        rationale = "Bull regime with positive momentum and room to run."
    elif regime == "Bear" or (rsi and rsi > 80):
        action = "HOLD with caution"
        rationale = "Elevated risk — consider tightening stop-loss."
    elif rsi and rsi < 30:
        action = "ACCUMULATE"
        rationale = "Oversold in current regime — potential recovery candidate."
    else:
        action = "HOLD"
        rationale = "Neutral signals — maintain position."

    lines = [
        f"**{symbol} Recommendation: {action}**",
        f"",
        f"Current regime: **{regime}** (confidence: {regime_confidence:.0%})",
        f"Current price: INR {current_price:,.2f}",
    ]

    if rsi is not None:
        rsi_status = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        lines.append(f"RSI (14-day): {rsi:.1f} ({rsi_status})")

    lines.extend([
        f"90-day Sharpe ratio: {sharpe_90d:.2f}",
        f"90-day annualized volatility: {vol_90d:.1%}",
        f"Last month return: {monthly_return:.2%}",
    ])

    if all_regime_stats and symbol in all_regime_stats.get(regime, {}):
        regime_ret = all_regime_stats[regime][symbol]
        lines.append(f"Historical {regime}-regime avg monthly return: {regime_ret:.2%}")

    lines.extend([
        f"",
        f"**Rationale:** {rationale}",
    ])

    if regime == "Bear":
        lines.append(f"Suggested stop-loss: INR {current_price * 0.92:,.2f} (8% below current)")
    elif regime == "Bull":
        lines.append(f"Suggested target: INR {current_price * 1.15:,.2f} (15% upside)")

    return "\n".join(lines)


def generate_portfolio_summary(portfolio, perf, regime):
    lines = [
        f"### {portfolio['name']} — {regime} Regime",
        f"",
        f"**Method:** {portfolio['method']}",
        f"**Expected annual return:** {perf['Annual_Return']:.2%}",
        f"**Annual volatility:** {perf['Annual_Volatility']:.2%}",
        f"**Sharpe ratio:** {perf['Sharpe_Ratio']:.2f}",
        f"**Sortino ratio:** {perf['Sortino_Ratio']:.2f}",
        f"**Maximum drawdown:** {perf['Max_Drawdown']:.2%}",
        f"**VaR (95%):** {perf['VaR_95']:.4f}",
        f"",
        f"**Top holdings:**",
    ]

    weights = portfolio["weights"]
    for symbol, weight in weights.nlargest(5).items():
        from .portfolio import get_sector
        sector = get_sector(symbol)
        lines.append(f"  - {symbol} ({sector}): {weight:.1%}")

    if regime == "Bear":
        lines.append(f"\n*Defensive portfolio — low-volatility, counter-cyclical sectors to preserve capital.*")
    elif regime == "Bull":
        lines.append(f"\n*Growth portfolio — high-momentum names with position limits for concentration risk.*")

    return "\n".join(lines)


def sector_rotation_analysis(returns, regimes):
    from .portfolio import get_sector

    stock_sectors = {col: get_sector(col) for col in returns.columns}
    sector_returns = {}

    for regime in regimes.unique():
        mask = regimes == regime
        reg_ret = returns.loc[mask]
        for sector in set(stock_sectors.values()):
            sector_stocks = [s for s, sec in stock_sectors.items() if sec == sector and s in returns.columns]
            if sector_stocks:
                avg_ret = reg_ret[sector_stocks].mean(axis=1).mean() * 252
                if sector not in sector_returns:
                    sector_returns[sector] = {}
                sector_returns[sector][regime] = avg_ret

    df = pd.DataFrame(sector_returns).T
    df.index.name = "Sector"
    return df.round(4)

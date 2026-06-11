import numpy as np
import pandas as pd


def _normalize_series(s):
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(50.0, index=s.index)
    return ((s - s.min()) / rng) * 100


def compute_return_score(returns, lookback_252=252, lookback_126=126):
    scores = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < lookback_126:
            scores[col] = np.nan
            continue

        window = min(lookback_252, len(r))
        total_ret = (1 + r.tail(window)).prod()
        cagr = total_ret ** (252 / window) - 1

        recent = (1 + r.tail(min(lookback_126, len(r)))).prod() - 1

        scores[col] = 0.6 * cagr + 0.4 * recent

    raw = pd.Series(scores).dropna()
    return _normalize_series(raw)


def compute_risk_score(returns):
    scores = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 60:
            scores[col] = np.nan
            continue

        ann_vol = r.std() * np.sqrt(252)

        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min()

        var95 = r.quantile(0.05)

        risk_raw = 0.4 * ann_vol + 0.4 * abs(dd) + 0.2 * abs(var95)
        scores[col] = -risk_raw

    raw = pd.Series(scores).dropna()
    return _normalize_series(raw)


def compute_momentum_score(stock_features):
    scores = {}
    for symbol, df in stock_features.items():
        if len(df) < 50:
            scores[symbol] = np.nan
            continue

        latest = df.iloc[-1]
        sub_scores = []

        if "RSI" in df.columns and not np.isnan(latest.get("RSI", np.nan)):
            rsi = latest["RSI"]
            rsi_score = max(0, 100 - 2.5 * ((rsi - 57.5) ** 2) / 10)
            sub_scores.append(rsi_score)

        if "MACD_Hist" in df.columns and not np.isnan(latest.get("MACD_Hist", np.nan)):
            macd_h = latest["MACD_Hist"]
            macd_score = min(100, max(0, 50 + macd_h * 500))
            sub_scores.append(macd_score)

        if "MA50" in df.columns and "Close" in df.columns:
            if latest["MA50"] > 0:
                pct_above = (latest["Close"] - latest["MA50"]) / latest["MA50"]
                ma_score = min(100, max(0, 50 + pct_above * 200))
                sub_scores.append(ma_score)

        if "MA200" in df.columns and "Close" in df.columns:
            if not np.isnan(latest.get("MA200", np.nan)) and latest["MA200"] > 0:
                pct_above_200 = (latest["Close"] - latest["MA200"]) / latest["MA200"]
                ma200_score = min(100, max(0, 50 + pct_above_200 * 150))
                sub_scores.append(ma200_score)

        scores[symbol] = np.mean(sub_scores) if sub_scores else np.nan

    raw = pd.Series(scores).dropna()
    return _normalize_series(raw)


def compute_stability_score(returns):
    scores = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 60:
            scores[col] = np.nan
            continue

        monthly = r.groupby(pd.Grouper(freq="M")).apply(lambda x: (1 + x).prod() - 1)
        pos_ratio = (monthly > 0).mean()

        rolling_ret = r.rolling(30).mean()
        ret_consistency = 1.0 / (rolling_ret.std() + 1e-6)

        scores[col] = 0.6 * pos_ratio + 0.4 * min(ret_consistency, 5) / 5

    raw = pd.Series(scores).dropna()
    return _normalize_series(raw)


def compute_volume_score(df_all):
    scores = {}
    for symbol in df_all["Symbol"].unique():
        sdf = df_all[df_all["Symbol"] == symbol].sort_values("Date")
        if len(sdf) < 60 or "Volume" not in sdf.columns:
            continue

        vol = sdf["Volume"].tail(252)
        close = sdf["Close"].tail(252)

        if vol.std() == 0 or len(vol) < 60:
            continue

        recent_vol = vol.tail(20).mean()
        avg_vol = vol.mean()
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        vp_corr = close.pct_change().corr(vol.pct_change())
        if np.isnan(vp_corr):
            vp_corr = 0

        raw_score = 0.5 * min(vol_ratio, 2.0) / 2.0 + 0.5 * (vp_corr + 1) / 2
        scores[symbol] = raw_score

    raw = pd.Series(scores).dropna()
    return _normalize_series(raw)


def compute_investment_scores(returns, stock_features, df_all):
    return_scores = compute_return_score(returns)
    risk_scores = compute_risk_score(returns)
    momentum_scores = compute_momentum_score(stock_features)
    stability_scores = compute_stability_score(returns)
    volume_scores = compute_volume_score(df_all)

    common = return_scores.index
    for s in [risk_scores, momentum_scores, stability_scores, volume_scores]:
        common = common.intersection(s.index)

    result = pd.DataFrame({
        "Return_Score": return_scores.reindex(common),
        "Risk_Score": risk_scores.reindex(common),
        "Momentum_Score": momentum_scores.reindex(common),
        "Stability_Score": stability_scores.reindex(common),
        "Volume_Score": volume_scores.reindex(common),
    })

    result["Investment_Score"] = (
        0.30 * result["Return_Score"] +
        0.25 * result["Risk_Score"] +
        0.20 * result["Momentum_Score"] +
        0.15 * result["Stability_Score"] +
        0.10 * result["Volume_Score"]
    ).round(1)

    def grade(score):
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 35:
            return "D"
        return "F"

    result["Grade"] = result["Investment_Score"].apply(grade)
    result = result.sort_values("Investment_Score", ascending=False)

    return result


def generate_score_explanation(symbol, scores):
    total = scores["Investment_Score"]
    grade = scores["Grade"]

    strengths = []
    weaknesses = []

    if scores["Return_Score"] >= 70:
        strengths.append("strong historical returns")
    elif scores["Return_Score"] < 40:
        weaknesses.append("weak return history")

    if scores["Risk_Score"] >= 70:
        strengths.append("low risk profile")
    elif scores["Risk_Score"] < 40:
        weaknesses.append("elevated risk (high volatility or deep drawdowns)")

    if scores["Momentum_Score"] >= 70:
        strengths.append("positive technical momentum")
    elif scores["Momentum_Score"] < 40:
        weaknesses.append("weak or negative momentum signals")

    if scores["Stability_Score"] >= 70:
        strengths.append("consistent return pattern")
    elif scores["Stability_Score"] < 40:
        weaknesses.append("inconsistent/erratic returns")

    if scores["Volume_Score"] >= 70:
        strengths.append("healthy volume activity")
    elif scores["Volume_Score"] < 40:
        weaknesses.append("declining or weak volume")

    parts = [f"{symbol} receives a score of {total:.0f}/100 (Grade: {grade})."]

    if strengths:
        parts.append(f"Strengths: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"Concerns: {', '.join(weaknesses)}.")

    if grade == "A":
        parts.append("This stock shows strong fundamentals across multiple dimensions.")
    elif grade == "B":
        parts.append("Solid overall profile with some areas for monitoring.")
    elif grade == "C":
        parts.append("Mixed signals — suitable for diversified portfolios but not a standalone pick.")
    elif grade in ("D", "F"):
        parts.append("Significant concerns — consider reducing exposure or avoiding new positions.")

    return " ".join(parts)

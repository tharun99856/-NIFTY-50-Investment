import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_all_stocks, get_stock, get_available_symbols, REPRESENTATIVE_STOCKS
from src.features import engineer_all_features
from src.regime import fit_regime_model, compute_index_returns, regime_summary, regime_transition_matrix
from src.models import run_prediction_pipeline, FEATURE_COLS_XGBOOST
from src.portfolio import (
    build_conservative_portfolio, build_balanced_portfolio, build_aggressive_portfolio,
    portfolio_performance, build_regime_portfolios, get_sector, compute_annual_stats,
    RISK_FREE_RATE,
)
from src.risk import compute_stock_risk_table, compute_portfolio_risk, detect_anomalies
from src.explainability import (
    generate_stock_recommendation, generate_portfolio_summary,
    sector_rotation_analysis,
)

st.set_page_config(page_title="NIFTY-50 Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .metric-card { background: #f0f2f6; border-radius: 10px; padding: 15px; margin: 5px; }
    h1 { color: #1f4e79; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return load_all_stocks()


@st.cache_data
def compute_features(df):
    symbols = get_available_symbols(df)
    featured = []
    for sym in symbols:
        stock = get_stock(df, sym)
        if len(stock) > 200:
            stock = engineer_all_features(stock)
            featured.append(stock)
    return pd.concat(featured, ignore_index=True)


@st.cache_data
def compute_regimes(df):
    index_ret = compute_index_returns(df)
    model, regimes = fit_regime_model(index_ret)
    summary = regime_summary(regimes, index_ret)
    return index_ret, regimes, summary, model


def main():
    st.title("NIFTY-50 Investment Intelligence Platform")
    st.markdown("---")

    try:
        raw_df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Place the Kaggle CSV files in `data/raw/` and reload.")
        st.code("pip install kagglehub\npython -c \"import kagglehub; kagglehub.dataset_download('rohanrao/nifty50-stock-market-data')\"")
        return

    df = compute_features(raw_df)
    symbols = get_available_symbols(df)
    index_ret, regimes, regime_summ, hmm_model = compute_regimes(raw_df)

    tab_eda, tab_regime, tab_predict, tab_portfolio, tab_risk, tab_explain = st.tabs([
        "📈 EDA", "🔄 Regime Detection", "🤖 Stock Predictor",
        "💼 Portfolio", "⚠️ Risk Assessment", "💡 Explainability"
    ])

    with tab_eda:
        st.header("Exploratory Data Analysis")

        col1, col2 = st.columns(2)
        with col1:
            top_stocks = [s for s in REPRESENTATIVE_STOCKS if s in symbols][:10]
            selected_eda = st.multiselect("Select stocks for analysis", symbols, default=top_stocks[:5])

        with col2:
            date_range = st.slider(
                "Date range",
                min_value=raw_df["Date"].min().to_pydatetime(),
                max_value=raw_df["Date"].max().to_pydatetime(),
                value=(raw_df["Date"].min().to_pydatetime(), raw_df["Date"].max().to_pydatetime()),
            )

        if selected_eda:
            mask = (df["Symbol"].isin(selected_eda)) & (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
            plot_df = df[mask]

            st.subheader("Price Trends")
            fig = px.line(plot_df, x="Date", y="Close", color="Symbol", title="Closing Price Over Time")
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Volume vs Price Correlation")
                for sym in selected_eda[:3]:
                    sdf = plot_df[plot_df["Symbol"] == sym]
                    if len(sdf) > 0:
                        corr = sdf["Close"].corr(sdf["Volume"])
                        st.metric(f"{sym}", f"{corr:.3f}")

            with col2:
                st.subheader("Rolling Volatility (30-day)")
                fig_vol = go.Figure()
                for sym in selected_eda[:5]:
                    sdf = plot_df[plot_df["Symbol"] == sym]
                    if "Rolling_Vol_30" in sdf.columns:
                        fig_vol.add_trace(go.Scatter(x=sdf["Date"], y=sdf["Rolling_Vol_30"],
                                                      name=sym, mode="lines"))
                fig_vol.update_layout(height=350, title="30-Day Rolling Volatility")
                st.plotly_chart(fig_vol, width="stretch")

            st.subheader("Return Distributions")
            fig_dist = make_subplots(rows=1, cols=min(3, len(selected_eda)),
                                      subplot_titles=selected_eda[:3])
            for i, sym in enumerate(selected_eda[:3]):
                sdf = plot_df[plot_df["Symbol"] == sym]
                if "Daily_Return" in sdf.columns:
                    returns = sdf["Daily_Return"].dropna()
                    fig_dist.add_trace(go.Histogram(x=returns, nbinsx=80, name=sym,
                                                      showlegend=False), row=1, col=i + 1)
            fig_dist.update_layout(height=350, title="Daily Return Distributions")
            st.plotly_chart(fig_dist, width="stretch")

            st.subheader("Correlation Heatmap")
            pivot_close = raw_df[raw_df["Symbol"].isin(selected_eda)].pivot_table(
                index="Date", columns="Symbol", values="Close"
            )
            corr_matrix = pivot_close.pct_change().corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  title="Return Correlation Matrix")
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, width="stretch")

    with tab_regime:
        st.header("Market Regime Detection")
        st.markdown("3-state HMM on NIFTY-50 returns: **Bull**, **Bear**, **Sideways**")

        pivot = raw_df.pivot_table(index="Date", columns="Symbol", values="Close")
        index_level = pivot.mean(axis=1)
        regime_df = pd.DataFrame({"Date": index_level.index, "Index": index_level.values})
        regime_df = regime_df.set_index("Date")
        regime_df = regime_df.join(regimes)

        fig_regime = go.Figure()
        colors = {"Bull": "rgba(0,200,0,0.15)", "Bear": "rgba(255,0,0,0.15)", "Sideways": "rgba(128,128,128,0.15)"}

        fig_regime.add_trace(go.Scatter(x=regime_df.index, y=regime_df["Index"],
                                         mode="lines", name="NIFTY-50 Index", line=dict(color="black", width=1.5)))

        for regime_name, color in colors.items():
            mask = regime_df["Regime"] == regime_name
            dates = regime_df.index[mask]
            if len(dates) > 0:
                for start, end in _get_contiguous_ranges(dates):
                    fig_regime.add_vrect(x0=start, x1=end, fillcolor=color,
                                          layer="below", line_width=0)

        fig_regime.update_layout(height=500, title="NIFTY-50 Index with Detected Market Regimes",
                                  xaxis_title="Date", yaxis_title="Index Level (avg close)")
        st.plotly_chart(fig_regime, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Regime Summary Statistics")
            st.dataframe(regime_summ.style.format("{:.4f}"), width="stretch")

        with col2:
            st.subheader("Regime Transition Matrix")
            trans = regime_transition_matrix(regimes)
            st.dataframe(trans.style.format("{:.3f}").background_gradient(cmap="YlOrRd"),
                         width="stretch")
        st.subheader("Regime Duration Distribution")
        from src.regime import get_regime_periods
        periods = get_regime_periods(regimes)
        fig_dur = px.histogram(periods, x="Days", color="Regime", barmode="overlay",
                                title="Duration of Regime Periods (days)")
        st.plotly_chart(fig_dur, width="stretch")

    with tab_predict:
        st.header("Stock Prediction Engine")
        st.markdown("Linear Regression baseline vs XGBoost with regime feature")

        pred_stocks = st.multiselect(
            "Select stocks to predict",
            symbols,
            default=[s for s in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"] if s in symbols][:3],
        )

        if st.button("Run Predictions", type="primary"):
            results = {}
            progress = st.progress(0)

            for i, sym in enumerate(pred_stocks):
                stock = df[df["Symbol"] == sym].copy()
                if len(stock) < 300:
                    st.warning(f"{sym}: insufficient data ({len(stock)} rows)")
                    continue

                stock = stock.set_index("Date")
                stock = stock.join(regimes)
                regime_map = {"Bull": 2, "Bear": 0, "Sideways": 1}
                stock["Regime_Num"] = stock["Regime"].map(regime_map).fillna(1)

                result = run_prediction_pipeline(stock, sym, regime_col="Regime_Num")
                results[sym] = result
                progress.progress((i + 1) / len(pred_stocks))

            if results:
                st.subheader("Model Comparison")
                metrics_rows = []
                for sym, r in results.items():
                    row = {"Symbol": sym}
                    for k, v in r["baseline_metrics"].items():
                        row[f"Baseline_{k}"] = v
                    for k, v in r["xgboost_metrics"].items():
                        row[f"XGBoost_{k}"] = v
                    metrics_rows.append(row)

                metrics_df = pd.DataFrame(metrics_rows)
                st.dataframe(metrics_df.style.format("{:.4f}", subset=metrics_df.columns[1:]),
                             width="stretch")

                for sym, r in results.items():
                    fig_pred = go.Figure()
                    test_dates = r["test_actual"].index
                    fig_pred.add_trace(go.Scatter(x=test_dates, y=r["test_actual"],
                                                    mode="lines", name="Actual", line=dict(color="black")))
                    fig_pred.add_trace(go.Scatter(x=r["xgboost_preds"].index, y=r["xgboost_preds"],
                                                    mode="lines", name="XGBoost", line=dict(color="blue", dash="dot")))
                    fig_pred.add_trace(go.Scatter(x=r["baseline_preds"].index, y=r["baseline_preds"],
                                                    mode="lines", name="Baseline", line=dict(color="red", dash="dash")))
                    fig_pred.update_layout(title=f"{sym} — Actual vs Predicted (Test Set)",
                                            height=400)
                    st.plotly_chart(fig_pred, width="stretch")

                    try:
                        import shap
                        xgb_model = r["xgboost_model"]
                        stock_data = df[df["Symbol"] == sym].copy().set_index("Date")
                        stock_data = stock_data.join(regimes)
                        regime_map = {"Bull": 2, "Bear": 0, "Sideways": 1}
                        stock_data["Regime_Num"] = stock_data["Regime"].map(regime_map).fillna(1)
                        feat_cols = [c for c in FEATURE_COLS_XGBOOST + ["Regime_Num"] if c in stock_data.columns]
                        X_sample = stock_data[feat_cols].dropna().tail(200)
                        explainer = shap.TreeExplainer(xgb_model)
                        shap_values = explainer.shap_values(X_sample)
                        importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
                        importance = importance.nlargest(10)

                        fig_shap = px.bar(x=importance.values, y=importance.index, orientation="h",
                                           title=f"{sym} — Feature Importance (SHAP)",
                                           labels={"x": "Mean |SHAP|", "y": "Feature"})
                        fig_shap.update_layout(height=350, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_shap, width="stretch")
                    except Exception:
                        pass

    with tab_portfolio:
        st.header("Regime-Aware Portfolio Construction")

        pivot_ret = raw_df.pivot_table(index="Date", columns="Symbol", values="Close").pct_change().iloc[1:]
        available = [s for s in pivot_ret.columns if pivot_ret[s].count() > 200]
        pivot_ret = pivot_ret[available]

        current_regime = regimes.iloc[-1] if len(regimes) > 0 else "Unknown"
        st.info(f"🔵 Current detected regime: **{current_regime}**")

        col1, col2, col3 = st.columns(3)
        portfolios = {}

        with col1:
            st.subheader("🛡️ Conservative")
            try:
                cons = build_conservative_portfolio(pivot_ret)
                cons_perf = compute_portfolio_risk(cons["weights"], pivot_ret)
                portfolios["Conservative"] = (cons, cons_perf)
                st.metric("Expected Return", f"{cons_perf['Annual_Return']:.2%}")
                st.metric("Volatility", f"{cons_perf['Annual_Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{cons_perf['Sharpe_Ratio']:.2f}")
                st.metric("Max Drawdown", f"{cons_perf['Max_Drawdown']:.2%}")
                st.markdown("**Top holdings:**")
                for sym, w in cons["weights"].nlargest(5).items():
                    st.write(f"  {sym} ({get_sector(sym)}): {w:.1%}")
            except Exception as e:
                st.error(f"Could not build: {e}")

        with col2:
            st.subheader("⚖️ Balanced")
            try:
                bal = build_balanced_portfolio(pivot_ret)
                bal_perf = compute_portfolio_risk(bal["weights"], pivot_ret)
                portfolios["Balanced"] = (bal, bal_perf)
                st.metric("Expected Return", f"{bal_perf['Annual_Return']:.2%}")
                st.metric("Volatility", f"{bal_perf['Annual_Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{bal_perf['Sharpe_Ratio']:.2f}")
                st.metric("Max Drawdown", f"{bal_perf['Max_Drawdown']:.2%}")
                st.markdown("**Top holdings:**")
                for sym, w in bal["weights"].nlargest(5).items():
                    st.write(f"  {sym} ({get_sector(sym)}): {w:.1%}")
            except Exception as e:
                st.error(f"Could not build: {e}")

        with col3:
            st.subheader("🚀 Aggressive")
            try:
                agg = build_aggressive_portfolio(pivot_ret)
                agg_perf = compute_portfolio_risk(agg["weights"], pivot_ret)
                portfolios["Aggressive"] = (agg, agg_perf)
                st.metric("Expected Return", f"{agg_perf['Annual_Return']:.2%}")
                st.metric("Volatility", f"{agg_perf['Annual_Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{agg_perf['Sharpe_Ratio']:.2f}")
                st.metric("Max Drawdown", f"{agg_perf['Max_Drawdown']:.2%}")
                st.markdown("**Top holdings:**")
                for sym, w in agg["weights"].nlargest(5).items():
                    st.write(f"  {sym} ({get_sector(sym)}): {w:.1%}")
            except Exception as e:
                st.error(f"Could not build: {e}")

        if portfolios:
            st.subheader("Sector Allocation Comparison")
            sector_data = {}
            for pname, (p, perf) in portfolios.items():
                sectors = p["weights"].groupby(p["weights"].index.map(get_sector)).sum()
                sector_data[pname] = sectors

            fig_sectors = go.Figure()
            all_sectors = sorted(set().union(*(s.index for s in sector_data.values())))
            for pname, sectors in sector_data.items():
                vals = [sectors.get(s, 0) for s in all_sectors]
                fig_sectors.add_trace(go.Bar(name=pname, x=all_sectors, y=vals))
            fig_sectors.update_layout(barmode="group", title="Sector Weights by Portfolio",
                                       yaxis_title="Weight", height=400)
            st.plotly_chart(fig_sectors, width="stretch")

        st.subheader("Regime-Conditional Portfolios")
        regime_ports = build_regime_portfolios(pivot_ret, regimes.reindex(pivot_ret.index).ffill().dropna())
        for regime_name, rp in regime_ports.items():
            with st.expander(f"{regime_name} Regime Portfolio"):
                rp_perf = compute_portfolio_risk(rp["weights"], pivot_ret)
                st.markdown(f"**Method:** {rp['method']}")
                for k, v in rp_perf.items():
                    if isinstance(v, float):
                        st.write(f"  {k}: {v:.4f}")
                st.markdown("**Holdings:**")
                for sym, w in rp["weights"].nlargest(5).items():
                    st.write(f"  {sym}: {w:.1%}")

    with tab_risk:
        st.header("Risk Analytics")

        pivot_ret_risk = raw_df.pivot_table(index="Date", columns="Symbol", values="Close").pct_change().iloc[1:]

        risk_stocks = st.multiselect(
            "Select stocks for risk analysis", symbols,
            default=[s for s in REPRESENTATIVE_STOCKS if s in symbols][:10],
            key="risk_stocks",
        )

        if risk_stocks:
            sel_ret = pivot_ret_risk[[s for s in risk_stocks if s in pivot_ret_risk.columns]]
            risk_table = compute_stock_risk_table(sel_ret, index_ret.reindex(sel_ret.index))
            st.subheader("Stock Risk Metrics")
            st.dataframe(
                risk_table.style.format("{:.4f}")
                .background_gradient(subset=["Sharpe_Ratio"], cmap="RdYlGn")
                .background_gradient(subset=["Max_Drawdown"], cmap="RdYlGn_r"),
                width="stretch",
            )

            st.subheader("Anomaly Detection (>3σ events)")
            anomaly_sym = st.selectbox("Stock for anomaly analysis", risk_stocks)
            if anomaly_sym and anomaly_sym in sel_ret.columns:
                returns = sel_ret[anomaly_sym].dropna()
                anomalies = detect_anomalies(returns)
                if len(anomalies) > 0:
                    stock_prices = raw_df[raw_df["Symbol"] == anomaly_sym].set_index("Date")["Close"]
                    fig_anom = go.Figure()
                    fig_anom.add_trace(go.Scatter(x=stock_prices.index, y=stock_prices,
                                                    mode="lines", name="Price", line=dict(color="steelblue")))
                    anom_dates = anomalies.index
                    anom_prices = stock_prices.reindex(anom_dates).dropna()
                    fig_anom.add_trace(go.Scatter(x=anom_prices.index, y=anom_prices,
                                                    mode="markers", name="Anomaly",
                                                    marker=dict(color="red", size=8)))
                    fig_anom.update_layout(title=f"{anomaly_sym} — Price with Anomalous Return Days",
                                            height=400)
                    st.plotly_chart(fig_anom, width="stretch")
                    st.caption(f"{len(anomalies)} anomalous days detected.")
                else:
                    st.info("No anomalies detected at 3σ threshold.")

            if portfolios:
                st.subheader("Portfolio Risk Comparison")
                port_risk_rows = []
                for pname, (p, perf) in portfolios.items():
                    port_risk_rows.append({"Portfolio": pname, **perf})
                port_risk_df = pd.DataFrame(port_risk_rows).set_index("Portfolio")
                st.dataframe(port_risk_df.style.format("{:.4f}"), width="stretch")

    with tab_explain:
        st.header("Explainability")

        col1, col2 = st.columns(2)
        with col1:
            explain_sym = st.selectbox("Select stock", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
        with col2:
            current_regime = regimes.iloc[-1] if len(regimes) > 0 else "Sideways"
            st.metric("Current Market Regime", current_regime)

        if explain_sym:
            stock = df[df["Symbol"] == explain_sym].copy()
            if len(stock) > 0:
                rec = generate_stock_recommendation(
                    explain_sym, stock, current_regime,
                    regime_confidence=0.78,
                )
                st.markdown(rec)
                st.markdown("---")

        st.subheader("Sector Rotation")
        pivot_ret_explain = raw_df.pivot_table(index="Date", columns="Symbol", values="Close").pct_change().iloc[1:]
        regime_aligned = regimes.reindex(pivot_ret_explain.index).ffill().dropna()
        sector_rot = sector_rotation_analysis(
            pivot_ret_explain.loc[regime_aligned.index],
            regime_aligned,
        )
        st.dataframe(sector_rot.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=None),
                     width="stretch")

        if portfolios:
            st.subheader("Portfolio Rationale")
            for pname, (p, perf) in portfolios.items():
                with st.expander(f"{pname} Portfolio"):
                    summary_text = generate_portfolio_summary(p, perf, current_regime)
                    st.markdown(summary_text)

    st.markdown("---")
    st.markdown("*Data: Kaggle rohanrao/nifty50-stock-market-data (Jan 2000 - Apr 2021)*")


def _get_contiguous_ranges(dates):
    if len(dates) == 0:
        return
    ranges = []
    start = dates[0]
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > 5:
            ranges.append((start, prev))
            start = d
        prev = d
    ranges.append((start, prev))
    return ranges


if __name__ == "__main__":
    main()

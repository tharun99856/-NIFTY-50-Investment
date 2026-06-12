# Working Prototype (Live Deployment)
https://niftyinvest.streamlit.app/
(Website might be slow, due to the limitations of free tier.)
# NIFTY-50 Investment Intelligence Platform

Investment analysis tool built on NIFTY-50 historical data (Jan 2000 - Apr 2021). Uses a Hidden Markov Model to detect market regimes (Bull/Bear/Sideways) and adapts predictions, portfolios, and risk metrics based on the current state.

## What it does

- **Regime Detection** - 3-state HMM on aggregate NIFTY-50 returns to classify market phases
- **Stock Predictor** - Linear Regression baseline + XGBoost with technical indicators and regime feature
- **Portfolio Construction** - Conservative (inverse-vol), Balanced (mean-variance), Aggressive (momentum-weighted)
- **Risk Assessment** - Sharpe, Sortino, Max Drawdown, VaR 95%, Beta for stocks and portfolios
- **Anomaly Detection** - Flags days with returns beyond 3 standard deviations
- **Explainability** - Stock recommendations and sector rotation analysis backed by computed metrics
- **Investment Score** - Composite 0-100 score (Return 30%, Risk 25%, Momentum 20%, Stability 15%, Volume 10%)

## Setup

```bash
git clone https://github.com/tharun99856/NIFTY-50-Investment.git
cd NIFTY-50-Investment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Dataset should already be in `data/raw/`. If not, download from:
https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data

Place all CSV files in `data/raw/`.

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with 6 tabs - EDA, Regime Detection, Stock Predictor, Portfolio, Risk Assessment, Explainability.

Notebooks are in `notebooks/` - run in order: 01_eda -> 02_predictor -> 03_portfolio -> 04_risk.

## Project Structure

```
├── data/raw/              # Kaggle CSV files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_predictor.ipynb
│   ├── 03_portfolio.ipynb
│   └── 04_risk.ipynb
├── src/
│   ├── data_loader.py     # loads and cleans CSV data
│   ├── features.py        # RSI, MACD, Bollinger, ATR, lag features etc.
│   ├── regime.py          # HMM fitting and regime labeling
│   ├── models.py          # Linear Regression + XGBoost pipelines
│   ├── portfolio.py       # 3 portfolio strategies + regime portfolios
│   ├── risk.py            # risk metrics and anomaly detection
│   ├── explainability.py  # recommendation generation
│   └── investment_score.py # composite investment scoring
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Features Used

Daily returns, log returns, MA20/50/200, EMA12/26, MACD + signal + histogram, RSI (14-day), Bollinger Bands (20-day, 2 std), 30-day rolling volatility, ATR (14-day), lag features (1, 2, 3, 5, 10 days).

## Models

- **Baseline**: Linear Regression on close_lag1-3, MA20, Volume
- **XGBoost**: 500 trees, max_depth=6, lr=0.05, uses all technical features + regime as numeric input
- Train/test split is time-based (last 20%), no shuffling

Evaluated using RMSE, MAE, R2, and directional accuracy. SHAP used for feature importance.

## Portfolio Strategies

| Strategy | Method | Constraints |
|----------|--------|-------------|
| Conservative | Inverse-volatility weighting | Max 10% per stock, 30% per sector |
| Balanced | Mean-variance optimization (max Sharpe) | Top stocks by Sharpe filtered |
| Aggressive | Momentum-weighted (6M cumulative return) | Max 20% per stock |

Regime-conditional portfolios also built - switches to defensive sectors (FMCG, Pharma, IT) during Bear regimes.

## Risk Metrics

- Annualized volatility (daily std * sqrt(252))
- Sharpe ratio (risk-free rate = 6%)
- Sortino ratio
- Maximum drawdown
- VaR at 95% confidence
- Beta vs equal-weighted NIFTY-50 index

## Dataset

- **Source**: [Kaggle - rohanrao/nifty50-stock-market-data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)
- **Period**: Jan 2000 - Apr 2021
- **Records**: ~235K rows, 50+ stocks across 10 sectors
- **Fields**: Date, Symbol, Open, High, Low, Close, Volume, Turnover

## Requirements

- Python 3.8+
- No GPU needed
- See `requirements.txt` for dependencies

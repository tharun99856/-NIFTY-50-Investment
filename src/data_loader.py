import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

REPRESENTATIVE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "BAJFINANCE", "SUNPHARMA", "TATAMOTORS", "LT", "WIPRO",
]


def load_all_stocks(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR

    skip_files = {"stock_metadata.csv", "NIFTY50_all.csv"}
    frames = []
    for f in os.listdir(data_dir):
        if not f.endswith(".csv") or f in skip_files:
            continue
        path = os.path.join(data_dir, f)
        try:
            file_df = pd.read_csv(path, parse_dates=["Date"])
        except (ValueError, KeyError):
            continue
        if "Symbol" not in file_df.columns:
            file_df["Symbol"] = f.replace(".csv", "")
        frames.append(file_df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. "
                                f"Download from kaggle: rohanrao/nifty50-stock-market-data")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    required = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in combined.columns:
            alt = [c for c in combined.columns if c.lower() == col.lower()]
            if alt:
                combined.rename(columns={alt[0]: col}, inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    if "Turnover" in combined.columns:
        combined["Turnover"] = pd.to_numeric(combined["Turnover"], errors="coerce")

    combined = combined.dropna(subset=["Close"])

    return combined


def get_stock(df, symbol):
    return df[df["Symbol"] == symbol].copy().sort_values("Date").reset_index(drop=True)


def get_available_symbols(df):
    return sorted(df["Symbol"].unique().tolist())

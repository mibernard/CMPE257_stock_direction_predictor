import pandas as pd
import requests
import time

def get_yahoo_history(symbol: str, interval='1d', lookback_days=730):
    """
    Fetch historical stock data directly from Yahoo Finance (no API key needed).
    """
    now = int(time.time())
    past = now - lookback_days * 24 * 60 * 60

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?symbol={symbol}&period1={past}&period2={now}&interval={interval}"
        f"&includePrePost=true&events=div%7Csplit%7Cearn"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    if "chart" not in data or data["chart"].get("error"):
        raise ValueError(f"Yahoo Finance returned error: {data['chart'].get('error')}")

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    indicators = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Datetime": pd.to_datetime(timestamps, unit="s"),
        "Open": indicators.get("open"),
        "High": indicators.get("high"),
        "Low": indicators.get("low"),
        "Close": indicators.get("close"),
        "Volume": indicators.get("volume"),
    })

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

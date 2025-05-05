import numpy as np
import pandas as pd

def add_technical_indicators(df):
    """Add technical indicators to the dataframe
    This is a direct port of the function from functions.py"""
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    weights = np.arange(1, 21)
    df['WMA'] = df['Close'].rolling(window=20).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df = df.dropna().reset_index(drop=True)
    return df 
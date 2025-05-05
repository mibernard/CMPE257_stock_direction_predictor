import pandas as pd
import lxml
import requests
import sys
import os

# Add the parent directory to the path so we can import YahooFinance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from YahooFinance import get_yahoo_history

class StockDataModel:
    @staticmethod
    def get_sp500_tickers():
        """Get list of S&P 500 tickers from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except Exception as e:
            print(f"Error loading S&P 500 tickers: {e}")
            # Return a small set of default tickers
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "V", "PG"]
        
    @staticmethod
    def get_stock_history(ticker, interval='1d', lookback_days=730):
        """Get stock history data"""
        try:
            return get_yahoo_history(ticker, interval=interval, lookback_days=lookback_days)
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None 
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Download 10 years of Apple (AAPL) stock price data
apple = yf.Ticker("AAPL")
df = apple.history(period="10y")

# Step 2: Save to CSV
df.to_csv("apple_stock_10y.csv")
print("Saved to apple_stock_10y.csv ✅")

# Step 3: (Optional) Plot the closing price
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["Close"], label="AAPL Close Price", color="blue")
plt.title("Apple (AAPL) Stock Price - Last 10 Years")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import functions
from YahooFinance import get_yahoo_history

def train_stock_day_classifier(ticker_symbol, feature_cols):
    # Step 1: Download LAST 2 years of stock price data — DAILY
    # stock = yf.Ticker(ticker_symbol)
    # df = stock.history(period="730d", interval="1h")
    # df.reset_index(inplace=True)
    df = get_yahoo_history(ticker_symbol, interval='1d', lookback_days=730)

    # Step 2: Create 'OnlyDate' and 'WeekNumber'
    df['OnlyDate'] = df['Datetime'].dt.date
    df['WeekNumber'] = (df['Datetime'].dt.isocalendar().week +
                        (df['Datetime'].dt.year - df['Datetime'].dt.year.min()) * 52)

    # Step 3: Get last Close price per week
    week_last_price = df.groupby('WeekNumber')['Close'].last()

    # Step 4: Generate labels
    labels = []
    week_numbers = sorted(week_last_price.index)

    for i in range(1, len(week_numbers)):
        prev_price = week_last_price.loc[week_numbers[i-1]]
        curr_price = week_last_price.loc[week_numbers[i]]
        pct_change = (curr_price - prev_price) / prev_price * 100
        if pct_change >= 10:
            label = 0
        elif 5 < pct_change < 10:
            label = 1
        elif -10 < pct_change < -5:
            label = 2
        elif pct_change <= -10:
            label = 3
        else:
            label = 4
        labels.append((week_numbers[i], label))

    labels_df = pd.DataFrame(labels, columns=['WeekNumber', 'Label'])

    # Step 5: Merge labels onto main df
    df = df.merge(labels_df, on='WeekNumber', how='inner')

    # Step 6: technical indicators
    df = functions.add_technical_indicators(df)

    # Step 7: Prepare X and y
    X = df[feature_cols]
    y = df['Label']

    # Step 8: Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, shuffle=True)

    print(X_train)
    # Step 9: Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 10: Validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Step 11: Test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Step 12: Return value
    return model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df

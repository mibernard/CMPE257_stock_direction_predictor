import yfinance as yf
import pandas as pd
import functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from YahooFinance import get_yahoo_history
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils.visualize import save_confusion_matrix


def train_stock_day_classifier(ticker_symbol, feature_cols, model_index):
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

    # ===============================================================
    # Step 9: Train model
    # ===============================================================
    model = None

    if model_index == "0":
        model = LogisticRegression(C=1.0)
    elif model_index == "1":
        model = SVC(C=1.0, kernel='rbf')
    elif model_index == "2":
        model = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    elif model_index == "3":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_index == "4":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    elif model_index == "5":
        model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    elif model_index == "6":
        model = GaussianNB(var_smoothing=1e-9)
    elif model_index == "7":
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    elif model_index == "8":
        model = BaggingClassifier(n_estimators=10)
    elif model_index == "9":
        model = ExtraTreesClassifier(n_estimators=100, max_depth=None)

    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 10: Validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Step 11: Test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    save_confusion_matrix(y_test, y_test_pred, filename='output/conf_matrix.png')
    # Step 12: Return value
    return model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df

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

def train_stock_hour_classifier(ticker_symbol, feature_cols, model_index):


    # Step 1: Download LAST 2 years of stock price data — HOURLY
    # stock = yf.Ticker(ticker_symbol)
    # df = stock.history(period="730d", interval="1h")
    # df.reset_index(inplace=True)
    df = get_yahoo_history(ticker_symbol, interval='1h', lookback_days=729)

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

    # Step 5: Merge labels
    df = df.merge(labels_df, on='WeekNumber', how='inner')

    # Step 6: Add technical indicators
    df = functions.add_technical_indicators(df)

    # Step 7: Windowed features
    X = []
    y = []
    window_size = 120  # 5 days * 24 hours

    for i in range(0, len(df) - window_size, window_size):
        window_data = df.iloc[i:i + window_size][feature_cols].values.flatten()
        X.append(window_data)

        if i + window_size < len(df):
            next_week_label = df.iloc[i + window_size]['Label']
            y.append(next_week_label)

    X = pd.DataFrame(X)
    y = pd.Series(y)

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

    # Step 10: Validation & Test
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    # ✅ Step 11: Return model & df for UI visualization
    save_confusion_matrix(y_test, y_test_pred, filename='output/conf_matrix.png')
    return model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df

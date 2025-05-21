import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

def train_stock_month_classifier(ticker_symbol, feature_cols, model_index):
    # Step 1: Download historical data - MONTHLY (increase lookback to 10 years for monthly data)
    # Use the consistent get_yahoo_history function instead of direct yf.download
    df = get_yahoo_history(ticker_symbol, interval='1mo', lookback_days=3650)  # ~10 years
    
    # Step 2: Create month period and filter full months
    df['Month'] = df['Datetime'].dt.to_period('M')
    df = df[df['Datetime'] == df.groupby('Month')['Datetime'].transform('max')]

    # Step 3: Get last Close price per month
    month_last_price = df.groupby('Month')['Close'].last()

    # Step 4: Generate labels
    labels = []
    months = sorted(month_last_price.index)

    for i in range(1, len(months)):
        prev_price = month_last_price[months[i-1]]
        curr_price = month_last_price[months[i]]
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
        labels.append((months[i], label))

    labels_df = pd.DataFrame(labels, columns=['Month', 'Label'])

    # Step 5: Merge labels onto main df
    # Convert Month to string format for merging
    df['Month'] = df['Month'].astype(str)
    labels_df['Month'] = labels_df['Month'].astype(str)

    # Now merge
    df = df.merge(labels_df, on='Month', how='inner')

    # Step 6: Add technical indicators
    df = functions.add_technical_indicators(df)

    # Step 7: Prepare X and y
    X = df[feature_cols]
    y = df['Label']
    
    # Print the number of samples for debugging
    print(f"Monthly analysis: {len(X)} samples after preprocessing")
    
    # Check if we have enough data
    if len(X) < 10:
        print(f"Warning: Only {len(X)} samples available for monthly analysis - results may be unreliable")

    # Step 8: Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    # Step 9: Split data with adjusted ratios for smaller datasets
    if len(X) < 30:  # For very small datasets
        # Use a simpler split - 70% train, 30% test (no validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, shuffle=True
        )
        # Make validation set equal to test set for consistency in return values
        X_val, y_val = X_test.copy(), y_test.copy()
    else:
        # Normal split for larger datasets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, shuffle=True
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )

    # ===============================================================
    # Step 10: Train model
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

    model.fit(X_train, y_train)

    # Step 11: Validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Step 12: Test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Return values in consistent order with day.py and hr.py
    save_confusion_matrix(y_test, y_test_pred, filename='output/conf_matrix.png')
    return model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df

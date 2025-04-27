import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Download the last 2 years of Apple (AAPL) stock price data — HOURLY
apple = yf.Ticker("AAPL")
df = apple.history(period="730d", interval="1h")
df.reset_index(inplace=True)

# Step 2: Create 'OnlyDate' and 'WeekNumber'
df['OnlyDate'] = df['Datetime'].dt.date
df['WeekNumber'] = (df['Datetime'].dt.isocalendar().week +
                    (df['Datetime'].dt.year - df['Datetime'].dt.year.min()) * 52)

# Step 3: Get last Close price per week (for the label)
week_last_price = df.groupby('WeekNumber')['Close'].last()

# Step 4: Generate labels for the next week
labels = []
week_numbers = sorted(week_last_price.index)

for i in range(1, len(week_numbers)):
    prev_price = week_last_price.loc[week_numbers[i - 1]]
    curr_price = week_last_price.loc[week_numbers[i]]
    pct_change = (curr_price - prev_price) / prev_price * 100
    if pct_change >= 10:
        label = 0
    elif pct_change > 5 and pct_change < 10:
        label = 1
    elif pct_change < -5 and pct_change > -10:
        label = 2
    elif pct_change < -10:
        label = 3
    else:
        label = 4
    labels.append((week_numbers[i], label))

# Step 5: Create labels DataFrame
labels_df = pd.DataFrame(labels, columns=['WeekNumber', 'Label'])

# Step 6: Merge labels onto main df
df = df.merge(labels_df, on='WeekNumber', how='inner')

# Step 7: Prepare data for 5-day window prediction
# Group data into 5-day windows
X = []
y = []

# Iterate through the dataset in 5-day chunks (120 hourly data points per 5-day window)
for i in range(0, len(df) - 120, 120):
    # Extract 5 days of hourly data (5 * 24 = 120 hours)
    window_data = df.iloc[i:i + 120][['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
    X.append(window_data)

    # The label for the next week (7 days after the last day in the window)
    next_week_label = df.iloc[i + 120]['Label']
    y.append(next_week_label)

# Convert to numpy arrays
X = pd.DataFrame(X)
y = pd.Series(y)
print(X.head())
# Step 8: Train/Validation/Test Split (70/10/20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2 / 3, random_state=42, shuffle=True)

# Step 9: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Validate the model (on validation set)
y_val_pred = model.predict(X_val)


# Step 11: Test the model (on test set)
y_test_pred = model.predict(X_test)

# Test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print(test_accuracy)
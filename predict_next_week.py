import pandas as pd

def predict_next_week_prices(model, df, feature_cols, window_size=120):
    if len(df) < window_size:
        raise ValueError("Not enough data for prediction.")

    result = []
    recent_df = df.copy()
    trading_hours_per_day = 7

    for day in range(7):
        for hour in range(trading_hours_per_day):
            recent_window = recent_df.iloc[-window_size:]
            X_new = recent_window[feature_cols].values.flatten().reshape(1, -1)

            predicted_price = model.predict(X_new)[0]
            result.append(predicted_price)

            next_row = {col: recent_df.iloc[-1][col] for col in feature_cols}
            next_row['Close'] = predicted_price

            if 'Datetime' in recent_df.columns:
                last_datetime = pd.to_datetime(recent_df.iloc[-1]['Datetime'])
                next_row['Datetime'] = last_datetime + pd.Timedelta(hours=1)

            recent_df = pd.concat([recent_df, pd.DataFrame([next_row])], ignore_index=True)

    return result

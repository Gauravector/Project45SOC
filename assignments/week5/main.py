# Importing required libraries

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Task 1: Data collection

def collect_stock_data():
    print("Task 1: Data Collection")
    
    # 1. Stock from input
    ticker = input("Enter Stock Ticker (e.g., AAPL, GOOGL, MARUTI.NS, TMCV.NS): ").strip().upper()
    if not ticker:
        print("Ticker cannot be empty. Defaulting to 'TMCV.NS'.")
        ticker = "TMCV.NS"    

    # 2. Date range from input
    start_date = input("Enter Start Date (YYYY-MM-DD) [Default 2025-01-01]: ").strip()
    if not start_date:
        start_date = "2025-01-01"
        
    end_date = input("Enter End Date (YYYY-MM-DD) [Default Today]: ").strip()
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # 3. Choose timeframe interval
    print("\nAvailable Timeframe Intervals:")
    print("1. Daily ('1d')")
    print("2. Weekly ('1wk')")
    print("3. Monthly ('1mo')")
    interval = input("Select interval choice (1/2/3) [Default 1]: ").strip()
    if not interval:
        interval = '1d'

    # fetch data using yf
    print(f"\nFetching data for {ticker} from {start_date} to {end_date} (Interval: {interval})...")
    try:
        stock_df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if stock_df.empty:
            print("No data found. Please double-check the ticker symbol and date format.")
            return None
        if isinstance(stock_df.columns, pd.MultiIndex):
            # Drops the Ticker level (GOOGL) and keeps just 'Close', 'High', 'Open', etc.
            stock_df.columns = stock_df.columns.get_level_values(0)
        print("Data successfully downloaded!")
        print(f"Total records retrieved: {len(stock_df)}")
        print("\n--- First 5 Rows of Retrieved Data ---")
        print(stock_df.head())
        
        return stock_df, ticker
        
    except Exception as e:
        print(f"❌ An error occurred during data download: {e}")
        return None
    
stock_data, symbol = collect_stock_data()

# Task 2: Data visualization

def plot_data(stock_data, symbol):
    print("\nTask 2: Data Visualization")
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Closing Price')
    plt.title(f'{symbol} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid()
    plt.show()

    stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['EMA_5'] = stock_data['Close'].ewm(span=5, adjust=False).mean()
    plt.figure(figsize=(12,6))
    plt.plot(stock_data.index, stock_data['Close'], label='Close', color='C0', linewidth=1)
    plt.plot(stock_data.index, stock_data['SMA_5'], label=f'SMA 5', color='C1', linewidth=1.5)
    plt.plot(stock_data.index, stock_data['EMA_5'], label=f'EMA 5', color='C2', linewidth=1.5)
    plt.title(f'{symbol} — Close with SMA5 & EMA5')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ema12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = ema12 - ema26
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['MACD'], label='MACD', linewidth=1.5)
    plt.plot(stock_data.index, stock_data['Signal'], label='Signal Line', linewidth=1.5)
    plt.title(f'{symbol} — MACD & Signal Line')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    close = stock_data['Close']
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    stock_data['RSI_14'] = rsi.fillna(50)

    plt.figure(figsize=(12, 4))
    plt.plot(stock_data.index, stock_data['RSI_14'], label=f'RSI ({14})', color='C0')
    plt.axhline(70, color='red', linestyle='--', linewidth=0.8)
    plt.axhline(30, color='green', linestyle='--', linewidth=0.8)
    plt.fill_between(rsi.index, 70, 100, color='red', alpha=0.1)
    plt.fill_between(rsi.index, 0, 30, color='green', alpha=0.1)
    plt.title(f'{symbol} RSI ({14})')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_data(stock_data, symbol)
stock_data.dropna(inplace=True)

# Task 3: Data preprocessing

def preprocess_data_for_lstm(stock_data, lookback_window=60):
    print("\nTask 3: Data Preprocessing & Sliding Window Sequences")
    
    price_col = 'Close'
        
    # Selecting features for LSTM model input
    feature_cols = [price_col, 'SMA_5', 'EMA_5', 'MACD', 'RSI_14']
    raw_features = stock_data[feature_cols].values
    
    # Normalizing Data using MinMaxScaler (scales data between 0 and 1)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(raw_features)
    target_scaler.fit(stock_data[[price_col]].values)
    
    print(f"Features normalized. Multi-dimensional array shape: {scaled_features.shape}")
   
    X, y = [], []
    
    for i in range(lookback_window, len(scaled_features)):
        X.append(scaled_features[i-lookback_window:i])
        y.append(scaled_features[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    print(f"Preprocessing Complete!")
    print(f"Input Feature Vector Shape (X): {X.shape} -> (Samples, Window Size, Feature Count)")
    print(f"Target Vector Shape (y): {y.shape} -> (Samples,)")
    
    return X, y, feature_scaler, target_scaler

X, y, feature_scaler, target_scaler = preprocess_data_for_lstm(stock_data)

# Task 4: Building and Training the LSTM Model

def build_and_train_lstm(X, y, epochs=20, batch_size=32):
    print("\nTask 4: Building and Training the LSTM Model")

    # train and test data split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    
    # defining LSTM input shape based on the training data
    timesteps = X_train.shape[1]
    features = X_train.shape[2]
    
    model.add(LSTM(units=50, return_sequences=False, input_shape=(timesteps, features)))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(model.summary())

    print(f"\nStarting training for {epochs} epochs...")
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
    print("Model training complete!")
    return model

model = build_and_train_lstm(X, y)

# Task 5: Model Evaluation and Prediction

def predict_and_forecast(model, X, y, stock_data, feature_scaler, target_scaler, lookback_window=60, forecast_steps=15):
    print("\nTask 5: Prediction, 15-Step Forecasting & Evaluation")
    
    train_predictions_scaled = model.predict(X)

    train_predictions = target_scaler.inverse_transform(train_predictions_scaled)
    actual_prices = target_scaler.inverse_transform(y.reshape(-1, 1))
    
    # Calculate R² score across training outputs
    r2 = r2_score(actual_prices, train_predictions)
    print(f"📊 Model Evaluation - Training Period R² Score: {r2:.4f}")

    current_batch = feature_scaler.transform(stock_data[['Close', 'SMA_5', 'EMA_5', 'MACD', 'RSI_14']].values[-lookback_window:])
    current_batch = current_batch.reshape((1, lookback_window, X.shape[2]))
    
    future_forecast_scaled = []
    
    for _ in range(forecast_steps):
        
        next_pred_scaled = model.predict(current_batch)[0, 0]
        future_forecast_scaled.append(next_pred_scaled)
        
        last_indicators = current_batch[0, -1, 1:] # Extract last scaled [EMA_5, MACD] values
        next_feature_row = np.hstack(([next_pred_scaled], last_indicators)).reshape(1, 1, X.shape[2])
        
        current_batch = np.append(current_batch[:, 1:, :], next_feature_row, axis=1)
        
    future_forecast = target_scaler.inverse_transform(np.array(future_forecast_scaled).reshape(-1, 1))
    
    plt.figure(figsize=(14, 7))
    
    df_index = stock_data.index[-len(actual_prices):]
    
    # Plot historical actual vs predicted training trends
    plt.plot(df_index, actual_prices, label='Actual Historical Prices', color='royalblue', linewidth=1.5)
    plt.plot(df_index, train_predictions, label=f'Model Fit Predictions (R²: {r2:.2f})', color='orange', linestyle='--', linewidth=1.2)
   
    last_date = df_index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=stock_data.index.freq or 'D')[1:]
    
    plt.plot(future_dates, future_forecast, label='15-Unit Future Forecast', color='crimson', marker='o', markersize=4, linewidth=1.5)
    
    plt.title("Stock Price Prediction & Future 15-Unit Forecast", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # for saving the different prediction charts as name of symbol and date-time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_forecast_{timestamp}.png"
    
    plt.savefig(filename, dpi=300)
    print(f"✅ Prediction charts successfully exported to '{filename}'")
    
    plt.show()
    return future_forecast


forecast = predict_and_forecast(model, X, y, stock_data, feature_scaler, target_scaler)
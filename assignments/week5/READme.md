# Stock Price Prediction with LSTM

## Introduction
This assignment builds a simple stock price forecasting pipeline using Python, Yahoo Finance data, and a Long Short-Term Memory (LSTM) neural network. The goal is to collect historical stock data, analyze it with technical indicators, train a deep learning model, and generate a short-term forecast for future prices.

## What this code does
The script performs the following steps:

1. Data Collection
   - Prompts the user for a stock ticker, start date, end date, and time interval.
   - Downloads historical market data using the yfinance library.
   - Validates the data and prepares it for analysis.

2. Data Visualization
   - Plots the closing price over time.
   - Adds moving averages (SMA and EMA) to show trend behavior.
   - Visualizes MACD and RSI indicators for technical analysis.

3. Data Preprocessing
   - Selects relevant features such as Close price, SMA, EMA, MACD, and RSI.
   - Scales the feature values using MinMaxScaler.
   - Creates sliding-window sequences for LSTM training.

4. LSTM Model Training
   - Splits the data into training and testing sets.
   - Builds an LSTM neural network using Keras.
   - Trains the model to learn patterns from historical stock sequences.

5. Prediction and Forecasting
   - Evaluates the model using the R² score.
   - Predicts historical values and compares them against actual prices.
   - Produces a short 15-step future forecast and saves the chart as an image.

## Expected results to look for
When you run the script, you should look for these outcomes:

- A successful download message showing the stock data was retrieved.
- Charts for closing price, moving averages, MACD, RSI, and predictions.
- A training log where the loss decreases during model training.
- A model evaluation score (R²) that indicates how well the predictions fit the past data.
- A forecast plot showing actual historical prices, model predictions, and future projected values.

## Notes
- The code is designed for educational use and demonstrates how LSTMs can be applied to time-series forecasting.
- Results may vary depending on the stock chosen, the date range, and the model training settings.
- The script saves a forecast chart image with a timestamp in the working folder.

## Requirements
Make sure the following Python libraries are installed:

- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow or keras

## How to run
Run the script using:

```bash
python main.py
```

You will be prompted to enter the stock ticker, date range, and time interval before the model begins training.

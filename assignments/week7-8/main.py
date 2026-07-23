import yfinance as yf
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import gymnasium as gym
from gymnasium import spaces
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
        print(f"An error occurred during data download: {e}")
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
    
    # Use the same split settings as the training phase to compute a proper held-out test R²
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    in_sample_predictions_scaled = model.predict(X)
    in_sample_predictions = target_scaler.inverse_transform(in_sample_predictions_scaled)
    actual_prices = target_scaler.inverse_transform(y.reshape(-1, 1))
    in_sample_r2 = r2_score(actual_prices, in_sample_predictions)

    test_predictions_scaled = model.predict(X_test)
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled)
    actual_test_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    test_r2 = r2_score(actual_test_prices, test_predictions)

    print(f"Model Evaluation - Test Period R² Score: {test_r2:.4f}")
    print(f"Model Fit (in-sample) R² Score: {in_sample_r2:.4f}")

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
    
    # Plot historical actual vs predicted training trends using full in-sample data
    plt.plot(df_index, actual_prices, label='Actual Historical Prices', color='royalblue', linewidth=1.5)
    plt.plot(df_index, in_sample_predictions, label=f'Model Fit Predictions (R²: {in_sample_r2:.2f})', color='orange', linestyle='--', linewidth=1.2)
   
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
    print(f"Prediction charts successfully exported to '{filename}'")
    
    plt.show()
    return future_forecast


forecast = predict_and_forecast(model, X, y, stock_data, feature_scaler, target_scaler)

#=================================================
#RL part
#=================================================
def split_data_for_6week_test(stock_data):
    print("\n=== Step 1: Isolating 6-Week Test Dataset ===")
    
    # 1. Identify the last date in the dataset
    last_date = stock_data.index[-1]
    
    # 2. Calculate the cutoff date (6 weeks / 42 days prior to last date)
    cutoff_date = last_date - timedelta(weeks=6)
    
    # 3. Split the DataFrame into training and testing sets
    train_data = stock_data[stock_data.index < cutoff_date].copy()
    test_data = stock_data[stock_data.index >= cutoff_date].copy()
    
    print(f"Total rows in full dataset: {len(stock_data)}")
    print(f"├── Training Set: {len(train_data)} rows ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"└── 6-Week Test Set: {len(test_data)} rows ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})")
    
    return train_data, test_data

# Split your dataset into training history and the 6-week battleground
train_data, test_data = split_data_for_6week_test(stock_data)

class StockTradingEnv(gym.Env):
    """A custom stock trading environment for Gymnasium"""
    
    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        # Keep a clean copy of the data without boundary NaNs
        self.df = df.dropna().reset_index(drop=True) 
        self.initial_balance = initial_balance
        
        # 1. Action Space: 0 = Hold, 1 = Buy (All in), 2 = Sell (All out)
        self.action_space = spaces.Discrete(3)
        
        # 2. Observation Space (What the agent sees)
        # We pass the same 5 features you built for the LSTM, plus 2 account tracking variables
        self.feature_cols = ['Close', 'SMA_5', 'EMA_5', 'MACD', 'RSI_14']
        self.num_features = len(self.feature_cols) + 2 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Restarts the environment for a new training episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        
        # Return the initial state observations
        obs = self._next_observation()
        return obs, {}
        
    def _next_observation(self):
        """Constructs the state array for the current day"""
        # Extract the technical indicators
        frame = self.df.loc[self.current_step, self.feature_cols].values
        
        # Append account information (Balance is divided by initial balance to scale the number down)
        obs = np.append(frame, [self.balance / self.initial_balance, self.shares_held])
        return obs.astype(np.float32)
        
    def step(self, action):
        """Executes a trade and steps the market forward one day"""
        current_price = self.df.loc[self.current_step, 'Close']
        prev_net_worth = self.net_worth
        
        # Execute the trading action
        if action == 1: # BUY
            shares_bought = self.balance // current_price
            if shares_bought > 0:
                self.shares_held += shares_bought
                self.balance -= shares_bought * current_price
                
        elif action == 2: # SELL
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                
        # Action 0 is HOLD (do nothing)

        # Move time forward by one day
        self.current_step += 1
        
        # Check if we ran out of historical data
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Calculate new net worth
        new_price = self.df.loc[self.current_step, 'Close'] if not terminated else current_price
        self.net_worth = self.balance + (self.shares_held * new_price)
        
        # 3. Reward Function: Did the action make or lose money?
        reward = self.net_worth - prev_net_worth
        
        # Fetch the next day's observations
        obs = self._next_observation() if not terminated else np.zeros(self.num_features, dtype=np.float32)
        info = {'net_worth': self.net_worth}
        
        return obs, reward, terminated, truncated, info

# Initialize the environment using your training data
env = StockTradingEnv(train_data)
print(f"Environment initialized! Observation shape: {env.observation_space.shape}")

class DQNAgent:
    """A Deep Q-Network Agent for Stock Trading"""
    
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size      # 7 features (Close, SMA_5, EMA_5, MACD, RSI_14, Cash, Shares)
        self.action_size = action_size    # 3 actions (Hold, Buy, Sell)
        
        # --- Hyperparameters ---
        self.memory = deque(maxlen=2000)   # Experience Replay Memory Buffer
        self.gamma = 0.95                  # Discount factor for future rewards
        self.epsilon = 1.0                 # Initial Exploration Rate (100% random actions at start)
        self.epsilon_min = 0.01            # Minimum Exploration Rate
        self.epsilon_decay = 0.995         # Exponential decay factor for exploration per episode
        self.learning_rate = 0.001         # Learning rate for Adam optimizer
        
        # Build the Q-Network
        self.model = self._build_model()

    def _build_model(self):
        """Builds a Dense Neural Network to predict Q-values for each action"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear') # Linear activation for Q-values
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores a trading step (experience) into memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Selects an action using an Epsilon-Greedy strategy"""
        # Explore: Choose a completely random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        # Exploit: Use the Q-Network to predict the best action
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        """Trains the Q-Network on a random sample of past experiences"""
        if len(self.memory) < batch_size:
            return # Wait until memory has enough experience steps
            
        # Sample a mini-batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch arrays for vectorization speed
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Predict current Q-values and future Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Bellman Equation update: Q(s, a) = r + gamma * max(Q(s', a'))
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target = rewards[i] + self.gamma * np.amax(next_q_values[i])
            targets[i][actions[i]] = target

        # Train the model on updated Q-targets
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay Epsilon (reduce randomness over time as agent learns)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

state_size = env.observation_space.shape[0]
agent = DQNAgent(state_size=state_size, action_size=3)
print(f"DQNAgent initialized! Input state dim: {state_size}, Neural Net Summary:")
print(agent.model.summary())

def train_dqn_agent(env, agent, episodes=20, batch_size=32):
    print("\n=== Step 4: Training the Deep Q-Network Agent ===")
    
    total_rewards = []
    final_net_worths = []
    
    for episode in range(1, episodes + 1):
        # 1. Reset the environment for a new trading run
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        
        # 2. Run through the training market data day-by-day
        while not terminated:
            # Select action using Epsilon-Greedy strategy
            action = agent.act(state)
            
            # Execute trade and step into the next day
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, terminated)
            
            # Move to the next day
            state = next_state
            episode_reward += reward
            
            # Train the Q-Network on a random sample from memory
            agent.replay(batch_size=batch_size)
            
        final_net_worth = info['net_worth']
        total_rewards.append(episode_reward)
        final_net_worths.append(final_net_worth)
        
        print(f"Episode {episode:02d}/{episodes} | "
              f"Ending Net Worth: ${final_net_worth:,.2f} | "
              f"Total Reward: ${episode_reward:,.2f} | "
              f"Epsilon: {agent.epsilon:.4f}")
              
    print("\nTraining complete! The agent has built its trading policy.")
    return agent, total_rewards, final_net_worths

# Train the agent over 20 episodes
agent, rewards, net_worths = train_dqn_agent(env, agent, episodes=20, batch_size=32)

def evaluate_and_compare_models(train_data, test_data, lstm_model, dqn_agent, feature_scaler, target_scaler, lookback_window=60):
    print("\n=== Step 5: Comparative Evaluation on Unseen 6-Week Test Set ===")
    
    # -------------------------------------------------------------------------
    # 1. RL Agent Evaluation (Exploitation Mode: Epsilon = 0.0)
    # -------------------------------------------------------------------------
    test_env = StockTradingEnv(test_data)
    dqn_agent.epsilon = 0.0  # Turn off random exploration
    
    obs, _ = test_env.reset()
    terminated = False
    
    rl_net_worth_history = [test_env.initial_balance]
    buy_signals = []
    sell_signals = []
    
    step_idx = 0
    while not terminated:
        current_price = test_env.df.loc[test_env.current_step, 'Close']
        action = dqn_agent.act(obs)
        
        if action == 1:   # Buy Action
            buy_signals.append((step_idx, current_price))
        elif action == 2: # Sell Action
            sell_signals.append((step_idx, current_price))
            
        obs, reward, terminated, truncated, info = test_env.step(action)
        rl_net_worth_history.append(info['net_worth'])
        step_idx += 1
        
    rl_final_worth = rl_net_worth_history[-1]
    rl_return_pct = ((rl_final_worth - test_env.initial_balance) / test_env.initial_balance) * 100
    
    # Calculate Risk: Maximum Drawdown (MDD)
    net_worth_series = pd.Series(rl_net_worth_history)
    running_max = net_worth_series.cummax()
    drawdown = (net_worth_series - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Buy & Hold Benchmark Return
    initial_price = test_data['Close'].iloc[0]
    final_price = test_data['Close'].iloc[-1]
    buy_hold_return_pct = ((final_price - initial_price) / initial_price) * 100
    
    # -------------------------------------------------------------------------
    # 2. LSTM Model Evaluation on Test Data
    # -------------------------------------------------------------------------
    # Prepend last 60 lookback days from training set to seamlessly predict test set
    combined_df = pd.concat([train_data.tail(lookback_window), test_data])
    feature_cols = ['Close', 'SMA_5', 'EMA_5', 'MACD', 'RSI_14']
    
    scaled_test_features = feature_scaler.transform(combined_df[feature_cols].values)
    
    X_test_seq, y_test_true = [], []
    for i in range(lookback_window, len(scaled_test_features)):
        X_test_seq.append(scaled_test_features[i-lookback_window:i])
        y_test_true.append(scaled_test_features[i, 0])
        
    X_test_seq = np.array(X_test_seq)
    y_test_true = np.array(y_test_true)
    
    lstm_test_pred_scaled = lstm_model.predict(X_test_seq)
    lstm_test_pred = target_scaler.inverse_transform(lstm_test_pred_scaled)
    actual_test_prices = target_scaler.inverse_transform(y_test_true.reshape(-1, 1))
    
    # Calculate R² Score
    r2 = r2_score(actual_test_prices, lstm_test_pred)
    
    # -------------------------------------------------------------------------
    # 3. Print Results Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*55)
    print("        6-WEEK TEST SET PERFORMANCE & RISK SUMMARY       ")
    print("="*55)
    print(f"LSTM Prediction R² Score     : {r2:.4f}")
    print(f"Buy & Hold Benchmark Return  : {buy_hold_return_pct:.2f}%")
    print(f"RL (DQN) Agent Return         : {rl_return_pct:.2f}%")
    print(f"RL Agent Max Drawdown (Risk)  : {max_drawdown:.2f}%")
    print("="*55)
    
    # -------------------------------------------------------------------------
    # 4. Multi-Plot Visualization
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    test_dates = test_data.index[:len(actual_test_prices)]
    
    # Top Subplot: Prices, LSTM Fits & RL Buy/Sell Actions
    ax1.plot(test_dates, actual_test_prices, label='Actual Close Price', color='royalblue', linewidth=1.8)
    ax1.plot(test_dates, lstm_test_pred, label=f'LSTM Forecast Curve (R²: {r2:.2f})', color='orange', linestyle='--', linewidth=1.5)
    
    # Plot RL Trading Signals
    if buy_signals:
        b_indices, b_prices = zip(*[(i, p) for i, p in buy_signals if i < len(test_dates)])
        ax1.scatter(test_dates[list(b_indices)], b_prices, color='green', marker='^', s=120, label='RL Buy Signal', zorder=5)
        
    if sell_signals:
        s_indices, s_prices = zip(*[(i, p) for i, p in sell_signals if i < len(test_dates)])
        ax1.scatter(test_dates[list(s_indices)], s_prices, color='red', marker='v', s=120, label='RL Sell Signal', zorder=5)
        
    ax1.set_title("6-Week Battleground: LSTM Predictions vs. RL Trading Actions", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price", fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom Subplot: Portfolio Growth vs Capital Baseline
    ax2.plot(test_data.index[:len(rl_net_worth_history)], rl_net_worth_history, label='RL Portfolio Net Worth', color='purple', linewidth=1.5)
    ax2.axhline(test_env.initial_balance, color='gray', linestyle=':', label='Initial Capital ($10,000)')
    ax2.set_title("RL Agent Portfolio Value & Risk Tracking", fontsize=12)
    ax2.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Timestamped plot export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nComparative visualization saved as '{filename}'")
    plt.show()

evaluate_and_compare_models(train_data, test_data, model, agent, feature_scaler, target_scaler)
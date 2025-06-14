import pandas as pd
import numpy as np
import gym
from fetchinfunc import fetch_stock_data, add_technical_indicators, preprocess_data
from RiskManagement import RiskManager
from TradingEnv import MultiStockTradingEnv
from Sentiment import SentimentEnhancedTradingEnv
from A2cAgent import A2CAgent, train_a2c_agent 

def run_complete_trading_system(stock_list, training_period, testing_period, algorithm='A2C', tune_hyperparameters=True):
    """
    Main function to fetch data, optionally tune, train the agent, and return results.
    This version correctly handles data scaling and hyperparameter tuning to prevent data leakage.
    """
    print("="*80)
    print("Starting Complete Trading System Workflow")
    print("="*80)

    # === Step 1: Fetch Data for the Entire Period (Train + Test) ===
    print("\n[Step 1/5] Fetching data...")
    full_stock_dfs = {}
    for ticker in stock_list:
        # Fetch data for the combined training and testing period
        df = fetch_stock_data(ticker, training_period[0], testing_period[1])
        if df is not None:
            # Add indicators to the full dataset before scaling
            full_stock_dfs[ticker] = add_simple_indicators(df)
        else:
            print(f"Failed to load data for {ticker}, it will be excluded.")

    if not full_stock_dfs:
        print("FATAL: No valid stock data found. Exiting.")
        return None

    # === Step 2: Fit Scaler on TRAINING DATA ONLY to Prevent Leakage ===
    print("\n[Step 2/5] Fitting data scaler on the training period to prevent data leakage...")
    # Identify feature columns to be scaled (all except the base price/volume columns)
    feature_cols = [col for col in next(iter(full_stock_dfs.values())).columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

    # Combine feature data from the training period ONLY
    combined_train_features = pd.concat([
        df.loc[training_period[0]:training_period[1]][feature_cols]
        for df in full_stock_dfs.values()
    ]).dropna()

    scaler = StandardScaler()
    scaler.fit(combined_train_features)
    print("Scaler fitted successfully on training data.")

    # Apply the fitted scaler to the entire dataset
    scaled_stock_dfs = {ticker: df.copy() for ticker, df in full_stock_dfs.items()}
    for ticker in scaled_stock_dfs:
        scaled_stock_dfs[ticker][feature_cols] = scaler.transform(full_stock_dfs[ticker][feature_cols])

    # Split the now-scaled data into train and test sets
    train_dfs = {ticker: df.loc[training_period[0]:training_period[1]] for ticker, df in scaled_stock_dfs.items()}
    test_dfs = {ticker: df.loc[testing_period[0]:testing_period[1]] for ticker, df in scaled_stock_dfs.items()}
    print("Data successfully scaled and split into training and testing sets.")

    # === Step 3: Hyperparameter Tuning (Optional) ===
    print("\n[Step 3/5] Setting up for Hyperparameter Tuning...")
    # Create a factory function that provides a fresh, fully enhanced environment for each tuning run
    def create_tuning_env():
        SentimentEnhancedEnv = SentimentEnhancedTradingEnv.enhance_environment(MultiStockTradingEnv)
        env = SentimentEnhancedEnv(
            stock_dfs=train_dfs,
            initial_balance=INITIAL_ACCOUNT_BALANCE,
            transaction_fee_percent=TRANSACTION_FEE_PERCENT,
            window_size=20
        )
        risk_manager = RiskManager()
        return risk_manager.integrate_with_trading_env(env, train_dfs)

    if tune_hyperparameters and algorithm.lower() == 'a2c':
        # Run the tuner to find the best hyperparameters on a small number of episodes
        best_params = tune_a2c_hyperparameters(create_tuning_env, episodes_per_experiment=5)
    else:
        # Use default parameters if not tuning
        print("Skipping hyperparameter tuning. Using default parameters.")
        best_params = {'actor_lr': 0.0002, 'critic_lr': 0.0004, 'gamma': 0.99}

    # === Step 4: Final Agent Training ===
    print("\n[Step 4/5] Preparing for final model training...")
    final_train_env = create_tuning_env()
    state_size = final_train_env.observation_space.shape
    action_size = final_train_env.action_space.n

    # Use the best parameters found from tuning (or defaults)
    agent = A2CAgent(state_size, action_size, **best_params)
    print(f"\nCreating final A2C agent with parameters: {best_params}")

    print("Starting FINAL agent training for 45 episodes...")
    training_rewards = train_a2c_agent(final_train_env, agent, episodes=45)

    # === Step 5: Returning Results ===
    print("\n[Step 5/5] Workflow complete. Returning trained agent, scaler, and test data.")
    return {
        "agent": agent,
        "scaler": scaler, # IMPORTANT: return the fitted scaler
        "test_dfs": test_dfs, # Pre-scaled test data
        "training_rewards": training_rewards,
        "best_params": best_params
    }

# Example usage
if __name__ == '__main__':
    # Define stocks to trade (Indian stocks)
    stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS',
          'ICICIBANK.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS']

    # Define time periods
    training_period = ("2018-01-01", "2022-01-01")
    testing_period = ("2022-01-02", "2022-12-31")

    # Run the complete system
    results = run_complete_trading_system(
        stocks,
        training_period,
        testing_period,
        algorithm='a2c'  # Options: 'dqn', 'a2c', 'ppo'
    )
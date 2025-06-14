# run_train_save.py

import os
import json
import joblib
import gym
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from fetchinfunc import fetch_stock_data, add_technical_indicators, preprocess_data
from RiskManagement import RiskManager
from TradingEnv import MultiStockTradingEnv
from Sentiment import SentimentEnhancedTradingEnv
from A2cAgent import A2CAgent, train_a2c_agent

# Constants
INITIAL_ACCOUNT_BALANCE = 1_00_000
TRANSACTION_FEE_PERCENT = 0.001

def run_complete_trading_system(stock_list, training_period, testing_period, algorithm='A2C', tune_hyperparameters=True):
    print("="*80)
    print("Starting Complete Trading System Workflow")
    print("="*80)

    # === Step 1: Fetch Data ===
    print("\n[Step 1/5] Fetching data...")
    full_stock_dfs = {}
    for ticker in stock_list:
        df = fetch_stock_data(ticker, training_period[0], testing_period[1])
        if df is not None:
            full_stock_dfs[ticker] = add_technical_indicators(df)
        else:
            print(f"Failed to load data for {ticker}, it will be excluded.")

    if not full_stock_dfs:
        print("FATAL: No valid stock data found. Exiting.")
        return None

    # === Step 2: Fit Scaler on Training Data Only ===
    print("\n[Step 2/5] Fitting scaler to training data...")
    feature_cols = [col for col in next(iter(full_stock_dfs.values())).columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    combined_train_features = pd.concat([
        df.loc[training_period[0]:training_period[1]][feature_cols]
        for df in full_stock_dfs.values()
    ]).dropna()

    scaler = StandardScaler()
    scaler.fit(combined_train_features)
    print("Scaler fitted.")

    scaled_stock_dfs = {ticker: df.copy() for ticker, df in full_stock_dfs.items()}
    for ticker in scaled_stock_dfs:
        scaled_stock_dfs[ticker][feature_cols] = scaler.transform(full_stock_dfs[ticker][feature_cols])

    train_dfs = {ticker: df.loc[training_period[0]:training_period[1]] for ticker, df in scaled_stock_dfs.items()}
    test_dfs = {ticker: df.loc[testing_period[0]:testing_period[1]] for ticker, df in scaled_stock_dfs.items()}
    print("Data scaling and splitting done.")

    # === Step 3: Hyperparameter Tuning ===
    print("\n[Step 3/5] Hyperparameter Tuning...")
    def create_tuning_env():
        SentimentEnhancedEnv = SentimentEnhancedTradingEnv.enhance_environment(MultiStockTradingEnv)
        env = SentimentEnhancedEnv(stock_dfs=train_dfs, initial_balance=INITIAL_ACCOUNT_BALANCE, transaction_fee_percent=TRANSACTION_FEE_PERCENT, window_size=20)
        return RiskManager().integrate_with_trading_env(env, train_dfs)

    if tune_hyperparameters and algorithm.lower() == 'a2c':
        from hyper_tuner import tune_a2c_hyperparameters  # Assume you have this
        best_params = tune_a2c_hyperparameters(create_tuning_env, episodes_per_experiment=5)
    else:
        print("Using default A2C parameters.")
        best_params = {'actor_lr': 0.0002, 'critic_lr': 0.0004, 'gamma': 0.99}

    # === Step 4: Final Training ===
    print("\n[Step 4/5] Training final agent...")
    final_env = create_tuning_env()
    state_size = final_env.observation_space.shape
    action_size = final_env.action_space.n

    agent = A2CAgent(state_size, action_size, **best_params)
    rewards = train_a2c_agent(final_env, agent, episodes=45)

    # === Step 5: Return Results ===
    return {
        "agent": agent,
        "scaler": scaler,
        "test_dfs": test_dfs,
        "training_rewards": rewards,
        "best_params": best_params
    }

def save_trading_model(results):
    os.makedirs('saved_models', exist_ok=True)

    agent_found = False
    agent = results.get('agent')
    scaler_to_save = results.get('scaler')

    if isinstance(agent, A2CAgent) and scaler_to_save:
        agent_found = True
        print("Agent and scaler located.")

    if agent_found:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"a2c_trading_model_{timestamp}"
        save_dir = 'saved_models'

        actor_path = os.path.join(save_dir, f"{model_name}_actor.weights.h5")
        agent.actor.save_weights(actor_path)
        print(f"Saved actor to: {actor_path}")

        critic_path = os.path.join(save_dir, f"{model_name}_critic.weights.h5")
        agent.critic.save_weights(critic_path)
        print(f"Saved critic to: {critic_path}")

        scaler_path = os.path.join(save_dir, f"{model_name}_scaler.joblib")
        joblib.dump(scaler_to_save, scaler_path)
        print(f"Saved scaler to: {scaler_path}")

        metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
        metadata = {
            "model_type": "A2C",
            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state_size": str(agent.state_size),
            "action_size": agent.action_size,
            "best_params": results.get("best_params", {}),
            "scaler_path": os.path.basename(scaler_path)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
        print("\n✅ Model, weights, scaler, and metadata saved successfully!")
    else:
        print("❌ Error: Agent or scaler missing in results.")

# === MAIN EXECUTION ===
if __name__ == '__main__':
    stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS',
              'ICICIBANK.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS']

    training_period = ("2018-01-01", "2022-01-01")
    testing_period = ("2022-01-02", "2022-12-31")

    results = run_complete_trading_system(stocks, training_period, testing_period, algorithm='a2c')
    if results:
        save_trading_model(results)

# backtester.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from fetchinfunc import fetch_stock_data, add_simple_indicators
from RiskManagement import RiskManager
from TradingEnv import MultiStockTradingEnv
from Sentiment import SentimentEnhancedTradingEnv
from A2cAgent import A2CAgent

# === Run Backtest Core Function ===
def run_backtest(env, agent, verbose=True):
    print("\n" + "="*20 + " Starting Backtest Run " + "="*20)

    state = env.reset()
    done = False

    portfolio_values = [env.portfolio_value]
    total_reward = 0
    actions_taken = []
    dates = []

    if hasattr(env, 'current_date'):
        dates.append(env.current_date)

    step_count = 0
    while not done:
        step_count += 1
        action, action_probs = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        actions_taken.append(action)

        if 'portfolio_value' in info:
            portfolio_values.append(info['portfolio_value'])
        if 'date' in info and info['date'] is not None:
            dates.append(info['date'])

        if verbose and step_count % 50 == 0:
            date_str = info.get('date', 'N/A')
            if hasattr(date_str, 'strftime'): date_str = date_str.strftime('%Y-%m-%d')
            print(f"  ... Step: {step_count}, Date: {date_str}, Portfolio Value: {info.get('portfolio_value', 0):.2f}")

    print("\n" + "="*20 + " Backtest Run Complete " + "="*20)

    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return_pct = (final_value / initial_value - 1) * 100 if initial_value > 0 else 0

    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdown = (rolling_max - portfolio_series) / rolling_max
    max_drawdown_pct = drawdown.max() * 100 if not drawdown.empty else 0

    if len(portfolio_values) > 1:
        daily_returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    print(f"\n--- Backtest Summary ---")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {total_return_pct:.2f}%")
    print(f"Maximum Drawdown:        {max_drawdown_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")

    if len(dates) == len(portfolio_values):
        plt.figure(figsize=(15, 7))
        plt.plot(dates, portfolio_values, label='Portfolio Value')
        plt.title('Agent Portfolio Value During Backtest', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'portfolio_values': portfolio_values,
        'total_reward': total_reward,
        'actions': actions_taken,
        'dates': dates
    }

# === Load Model and Run Backtest ===
def load_and_backtest_a2c_model(actor_path, critic_path, tickers, start_date, end_date, initial_balance=1_000_000, transaction_fee=0.001):
    print("="*80)
    print("--- Starting Backtest Workflow ---")
    print("="*80)

    print(f"\n[Step 1/4] Loading metadata and scaler...")
    metadata_path = actor_path.replace('_actor.weights.h5', '_metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        original_state_size = eval(metadata['state_size'])
        original_action_size = metadata['action_size']
        best_params = metadata.get('best_params', {'actor_lr': 0.0002, 'critic_lr': 0.0004, 'gamma': 0.99})

        scaler_path = os.path.join(os.path.dirname(actor_path), metadata['scaler_path'])
        scaler = joblib.load(scaler_path)
        print("Scaler loaded.")
    except Exception as e:
        print(f"FATAL: Could not load metadata or scaler. {e}")
        return None

    print("\n[Step 2/4] Fetching and preprocessing backtest data...")
    backtest_dfs_raw = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            backtest_dfs_raw[ticker] = add_simple_indicators(df)
    if not backtest_dfs_raw:
        print("FATAL: No valid stock data.")
        return None

    feature_cols = [col for col in next(iter(backtest_dfs_raw.values())).columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    backtest_dfs_scaled = {ticker: df.copy() for ticker, df in backtest_dfs_raw.items()}
    for ticker, df in backtest_dfs_scaled.items():
        df[feature_cols] = scaler.transform(df[feature_cols])
    print("Data scaled.")

    print("\n[Step 3/4] Creating environment...")
    SentimentEnhancedEnv = SentimentEnhancedTradingEnv.enhance_environment(MultiStockTradingEnv)
    env = SentimentEnhancedEnv(
        stock_dfs=backtest_dfs_scaled,
        initial_balance=initial_balance,
        transaction_fee_percent=transaction_fee,
        window_size=original_state_size[0]
    )

    print("\n[Step 4/4] Loading agent and running backtest...")
    agent = A2CAgent(state_size=original_state_size, action_size=original_action_size, **best_params)
    agent.actor.load_weights(actor_path)
    agent.critic.load_weights(critic_path)
    print("Model weights loaded.")

    return run_backtest(env, agent)

# === Run Backtest on Latest Model if Executed Directly ===
if __name__ == '__main__':
    # === CONFIG ===
    stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS',
              'ICICIBANK.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS']
    start_date = '2022-01-02'
    end_date = '2022-12-31'
    model_dir = 'saved_models'

    model_files = [f for f in os.listdir(model_dir) if f.endswith('_actor.weights.h5')]
    if not model_files:
        print("❌ No saved models found.")
        exit(1)

    latest_actor_file = sorted(model_files)[-1]
    model_base = latest_actor_file.replace('_actor.weights.h5', '')
    actor_path = os.path.join(model_dir, latest_actor_file)
    critic_path = os.path.join(model_dir, f"{model_base}_critic.weights.h5")

    if os.path.exists(actor_path) and os.path.exists(critic_path):
        print(f"✅ Found latest model: {model_base}")
        backtest_results = load_and_backtest_a2c_model(
            actor_path=actor_path,
            critic_path=critic_path,
            tickers=stocks,
            start_date=start_date,
            end_date=end_date,
            initial_balance=1_000_000,
            transaction_fee=0.001
        )
    else:
        print("❌ Could not find both actor and critic weights.")

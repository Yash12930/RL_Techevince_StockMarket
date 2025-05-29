import pandas as pd
import numpy as np
import gym
from fetchinfunc import fetch_stock_data, add_technical_indicators, preprocess_data
from RiskManagement import RiskManager
from TradingEnv import MultiStockTradingEnv
from Sentiment import SentimentEnhancedTradingEnv
from A2cAgent import A2CAgent, train_a2c_agent 

def run_complete_trading_system(stock_list, training_period, testing_period, algorithm='ppo'):
    print("Starting trading system with all enhancements...")
    print(f"Stocks: {stock_list}")
    print(f"Training period: {training_period[0]} to {training_period[1]}")
    print(f"Testing period: {testing_period[0]} to {testing_period[1]}")
    print(f"Algorithm: {algorithm.upper()}")

    # 1. Fetch and process data using imported functions
    stock_dfs = {}
    for ticker in stock_list:
        df = fetch_stock_data(ticker, training_period[0], testing_period[1])
        if df is not None and not df.empty:
            df_processed = add_technical_indicators(df)
            df_final = preprocess_data(df_processed)
            stock_dfs[ticker] = df_final
        else:
            print(f"⚠️ Warning: No data for {ticker}")
    training_dfs = {}
    testing_dfs = {}

    for ticker, df in stock_dfs.items():
        train_df = df.loc[df.index >= training_period[0]]
        train_df = train_df.loc[train_df.index <= training_period[1]].copy()

        test_df = df.loc[df.index >= testing_period[0]]
        test_df = test_df.loc[test_df.index <= testing_period[1]].copy()

        if train_df.empty or test_df.empty:
            print(f"⚠️ Warning: {ticker} has empty training or testing data.")
            continue

        training_dfs[ticker] = train_df
        testing_dfs[ticker] = test_df

    if not training_dfs or not testing_dfs:
        raise ValueError("❌ No valid stock data available for training/testing.")

    base_env_class = MultiStockTradingEnv
    sentiment_env_class = SentimentEnhancedTradingEnv.enhance_environment(base_env_class)
    sentiment_env = sentiment_env_class(stock_dfs=training_dfs)
    risk_manager = RiskManager(max_position_size=0.2, stop_loss_pct=0.05, max_drawdown=0.15)
    risk_env = risk_manager.integrate_with_trading_env(sentiment_env, training_dfs)

    state_size = risk_env.observation_space.shape[0]
    action_size = risk_env.action_space.n


    if algorithm.lower() == 'a2c':
        agent = A2CAgent(state_size, action_size)
        train_fn = train_a2c_agent


    print(f"\nTraining {algorithm.upper()} agent...")
    training_results = train_fn(risk_env, agent)
    print("\nEvaluating agent on test data...")
    test_risk_env = risk_manager.integrate_with_trading_env(
        SentimentEnhancedTradingEnv.enhance_environment(MultiStockTradingEnv)(stock_dfs=testing_dfs),
        testing_dfs
    )

    state, _ = test_risk_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, info = test_risk_env.step(action)
        state = next_state
        total_reward += reward

    test_risk_env.render()

    initial_value = 100000 
    final_value = initial_value + total_reward  
    profit = final_value - initial_value
    roi = (profit / initial_value) * 100 if initial_value != 0 else 0

    print(f"\nPerformance Summary:")
    print(f"Initial portfolio value: ₹{initial_value:.2f}")
    print(f"Final portfolio value: ₹{final_value:.2f}")
    print(f"Profit/Loss: ₹{profit:.2f}")
    print(f"ROI: {roi:.2f}%")

    # Buy & Hold Comparison
    print("\nComparison with Buy & Hold strategy:")
    total_bh_return = 0
    valid_tickers = 0

    for ticker in stock_list:
        if ticker not in testing_dfs or testing_dfs[ticker].empty:
            print(f"Skipping {ticker} due to missing data.")
            continue

        first_price = testing_dfs[ticker].iloc[0]['Close']
        last_price = testing_dfs[ticker].iloc[-1]['Close']

        if pd.isna(first_price) or pd.isna(last_price):
            print(f"Skipping {ticker} due to NaN values in Close prices.")
            continue

        bh_return = (last_price - first_price) / first_price * 100 if first_price != 0 else 0
        total_bh_return += bh_return
        valid_tickers += 1

        print(f"{ticker}: {bh_return:.2f}%")

    if valid_tickers > 0:
        avg_bh_return = total_bh_return / valid_tickers
    else:
        avg_bh_return = 0

    print(f"Average Buy & Hold return: {avg_bh_return:.2f}%")
    print(f"Algorithm outperformance: {roi - avg_bh_return:.2f}%")

    return {
        'agent': agent,
        'env': test_risk_env,
        'initial_value': initial_value,
        'final_value': final_value,
        'roi': roi,
        'buy_hold_roi': avg_bh_return
    }

if __name__ == '__main__':
    stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]
    training_period = ("2018-01-01", "2022-01-01")
    testing_period = ("2022-01-02", "2022-12-31")

    results = run_complete_trading_system(
        stocks,
        training_period,
        testing_period,
        algorithm='ppo'
    )

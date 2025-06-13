import numpy as np



class RiskManager:
    """
    Risk management wrapper for trading environments
    """
    def __init__(self, max_position_size=0.2, stop_loss_pct=0.2, max_drawdown=0.15):
        self.max_position_size = max_position_size  # Maximum size of any position as % of portfolio
        self.stop_loss_pct = stop_loss_pct  # Stop loss percentage
        self.max_drawdown = max_drawdown  # Maximum allowed drawdown

    def integrate_with_trading_env(self, env, stock_dfs):
        """
        Wrap the trading environment with risk management
        """
        # Store entry prices for each position for stop loss tracking
        env.entry_prices = {symbol: 0 for symbol in env.symbols}
        env.max_portfolio_value = env.portfolio_value

        # Modify the step method to include risk management
        original_step = env.step

        def risk_managed_step(action):
    # Check if the action would exceed max position size
            if not env.done and action < env.action_space.n - 1:  # Not the rebalance action
                stock_idx = action // 9
                action_type = action % 9
                symbol = env.symbols[stock_idx]

                if action_type >= 1 and action_type <= 4:  # Buy actions
                    # Calculate the size of the position after buying
                    current_price = env.stock_dfs[symbol].loc[env.current_date]['Close']
                    shares = env.portfolio[symbol]
                    current_value = shares * current_price
                    portfolio_value = env.balance + env._calculate_stocks_value()

                    # Determine amount to buy based on action_type (25%, 50%, 75%, 100%)
                    buy_percentage = action_type * 0.25
                    amount_to_buy = env.balance * buy_percentage

                    # Check if this would exceed the max position size
                    new_position_value = current_value + amount_to_buy
                    new_position_pct = new_position_value / portfolio_value

                    if new_position_pct > self.max_position_size:
                        # Adjust to max position size
                        amount_to_buy = (self.max_position_size * portfolio_value) - current_value
                        # If already at or exceeding max position, take no action
                        if amount_to_buy <= 0:
                            action = env.action_space.n - 1  # Change to "no action"

                # Track entry prices for stop losses
                if action_type >= 1 and action_type <= 4 and env.entry_prices[symbol] == 0:
                    env.entry_prices[symbol] = current_price

            # Execute the original step
            state, reward, done, info = original_step(action)
            env.done = done  # Make sure the done flag is properly set

            # Check for stop losses
            for symbol in env.symbols:
                if env.portfolio[symbol] > 0:
                    current_price = env.stock_dfs[symbol].loc[env.current_date]['Close']
                    entry_price = env.entry_prices[symbol]

                    # If we've lost more than stop_loss_pct, sell the position
                    if entry_price > 0 and (current_price / entry_price - 1) < -self.stop_loss_pct:
                        # Create a sell action for this stock
                        stock_idx = env.symbols.index(symbol)
                        sell_action = stock_idx * 9 + 8  # Sell 100%

                        # Execute the sell
                        state, reward, done, info = original_step(sell_action)
                        env.entry_prices[symbol] = 0  # Reset entry price

            # Check for max drawdown
            env.max_portfolio_value = max(env.max_portfolio_value, env.portfolio_value)
            drawdown = (env.max_portfolio_value - env.portfolio_value) / env.max_portfolio_value

            if drawdown > self.max_drawdown:
                # Sell all positions
                for symbol in env.symbols:
                    if env.portfolio[symbol] > 0:
                        stock_idx = env.symbols.index(symbol)
                        sell_action = stock_idx * 9 + 8  # Sell 100%
                        state, reward, done, info = original_step(sell_action)
                        env.entry_prices[symbol] = 0

            return state, reward, done, info

        # Replace the step method
        env.step = risk_managed_step
        return env

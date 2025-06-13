import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MultiStockTradingEnv(gym.Env):
    """
    Trading environment that handles multiple stocks simultaneously
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_dfs, initial_balance=1000000, transaction_fee_percent=0.0001,
                window_size=20):
        super(MultiStockTradingEnv, self).__init__()

        self.stock_dfs = stock_dfs  # Dictionary of dataframes {symbol: dataframe}
        self.symbols = list(stock_dfs.keys())
        self.n_stocks = len(self.symbols)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent

        # Find common dates across all stocks
        common_dates = None
        for symbol, df in stock_dfs.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        self.common_dates = sorted(list(common_dates))
        print(f"Trading environment created with {len(self.common_dates)} common trading days")

        # Action space: For each stock: [no action, buy 25%, buy 50%, buy 75%, buy 100%, sell 25%, sell 50%, sell 75%, sell 100%]
        # Plus one action for rebalancing the portfolio
        self.action_space = spaces.Discrete(self.n_stocks * 9 + 1)

        # Observation space: window_size of data for each stock + portfolio state
        self.feature_count = len(next(iter(stock_dfs.values())).columns)  # Features per stock
        portfolio_features = 2 + self.n_stocks  # Cash + total value + allocation per stock

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_stocks * self.feature_count + portfolio_features),
            dtype=np.float32
        )

        self.position_cost = {symbol: PositionTracker() for symbol in self.symbols}
        self.win_rate = 0.5       # Initial win probability
        self.win_loss_ratio = 1   # Initial win/loss ratio

        # Initialize episode variables
        self.reset()

    def reset(self):
        """Reset the environment to the initial state"""
        # Reset portfolio
        self.portfolio = {symbol: 0 for symbol in self.symbols}
        self.balance = self.initial_balance

        # Find common dates across all stocks if not already defined
        if not hasattr(self, 'common_dates'):
            # Find common dates across all stocks
            self.common_dates = sorted(set.intersection(
                *[set(df.index) for df in self.stock_dfs.values()]
            ))
            if not self.common_dates:
                raise ValueError("No common dates found across stocks")

        # Reset current step
        self.window_size = min(self.window_size, len(self.common_dates) - 1)  # Safety check
        self.current_date_idx = self.window_size

        # Set current date based on index
        if 0 <= self.current_date_idx < len(self.common_dates):
            self.current_date = self.common_dates[self.current_date_idx]
        else:
            # Fallback if index is invalid
            self.current_date = self.common_dates[0] if self.common_dates else None
            self.current_date_idx = 0

        print(f"Reset environment: current_date_idx={self.current_date_idx}, current_date={self.current_date}")

        # Reset tracking variables
        self.portfolio_value_history = [self.initial_balance]
        self.trade_history = {symbol: [] for symbol in self.symbols}

        # Reset episode state
        self.done = False  # Initialize done flag

        # Calculate initial portfolio value
        try:
            stock_value = self._calculate_stocks_value()
            self.portfolio_value = self.balance + stock_value
        except Exception as e:
            print(f"Error calculating portfolio value during reset: {e}")
            self.portfolio_value = self.balance

        self.portfolio_value_history.append(self.portfolio_value)

        # Get initial observation
        try:
            observation = self._get_observation()
        except Exception as e:
            print(f"Error getting observation during reset: {e}")
            # Create a placeholder observation if there's an error
            observation = np.zeros((self.window_size, len(self.symbols) * 5 + 2))

        return observation

    def step(self, action):
        """Take an action in the environment using multi-stock logic"""
        # Check if the episode is already marked as done
        if self.done:
            # Return the last valid observation or a zero observation if called after done
            last_observation = self._get_observation() if hasattr(self, 'current_date') else np.zeros(self.observation_space.shape, dtype=np.float32)
            print("Warning: step() called after episode finished. Returning last state.")
            return last_observation, 0, True, {"portfolio_value": self.portfolio_value}

        # Initialize sentiment_debug_info at the beginning of the method
        sentiment_debug_info = {'available': False}  # Default value

        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value
        total_transaction_cost = 0  # Initialize cost for this step

        # Execute action and get trade reward/cost
        try:
            trade_reward, transaction_cost = self._take_action(action)
            total_transaction_cost += transaction_cost
        except Exception as e:
            print(f"Error during _take_action for action {action}: {e}")
            trade_reward = -10  # Penalize if action execution fails
            transaction_cost = 0
            total_transaction_cost = 0

        # Move to the next date index
        self.current_date_idx += 1

        # Check if the episode has ended AFTER incrementing the index
        self.done = self.current_date_idx >= len(self.common_dates)

        # Initialize variables for this scope
        new_portfolio_value = self.portfolio_value  # Default to previous value if done
        next_observation = None

        if not self.done:
            # Update current date ONLY if not done
            try:
                self.current_date = self.common_dates[self.current_date_idx]
            except IndexError:
                print(f"Error: current_date_idx {self.current_date_idx} is out of bounds for common_dates (len: {len(self.common_dates)}). Ending episode.")
                self.done = True
                # Use the last valid date from the list for final calculations if possible
                self.current_date = self.common_dates[-1] if self.common_dates else None

            # Calculate new portfolio value AFTER updating date and executing trades
            try:
                # Ensure current_date is valid before calculating value
                if self.current_date:
                    new_portfolio_value = self.balance + self._calculate_stocks_value()
                    self.portfolio_value = new_portfolio_value  # Update the main portfolio value attribute
                    self.portfolio_value_history.append(new_portfolio_value)
                else:
                    # If date is invalid, keep previous value
                    new_portfolio_value = prev_portfolio_value
            except Exception as e:
                print(f"Error calculating portfolio value on {self.current_date}: {e}. Using previous value.")
                new_portfolio_value = prev_portfolio_value  # Revert to previous value on error

            # Get the next observation based on the NEW date
            try:
                # Ensure current_date is valid before getting observation
                if self.current_date:
                    next_observation = self._get_observation()
                else:
                    # If date is invalid, return zero observation and end
                    next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                    self.done = True
            except Exception as e:
                print(f"Error getting observation on {self.current_date}: {e}. Returning zero observation.")
                next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                self.done = True  # Consider ending episode if observation fails

        else:
            # If done at the start of the step (or became done due to index increment)
            # Use the last calculated portfolio value
            new_portfolio_value = self.portfolio_value
            # Create a final zero observation as the terminal state representation
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ================== Enhanced Normalized Reward Calculation ==================
        # Initialize raw reward components
        raw_components = {
            'portfolio_change': 0,
            'sharpe_ratio': 0,
            'drawdown': 0,
            'sentiment_alignment': 0
        }

        # Calculate raw components
        # 1. Portfolio Value Change
        if prev_portfolio_value > 0:
            portfolio_change = (new_portfolio_value / prev_portfolio_value) - 1
            raw_components['portfolio_change'] = portfolio_change * 100  # Convert to percentage

        # 2. Sharpe Ratio
        if len(self.portfolio_value_history) > 30:
            returns = np.diff(np.log(self.portfolio_value_history[-30:]))
            if len(returns) > 1 and np.std(returns) > 1e-9:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                raw_components['sharpe_ratio'] = sharpe

        # 3. Drawdown
        if len(self.portfolio_value_history) > 1:
            peak = np.max(self.portfolio_value_history)
            if peak > 0:
                current_drawdown = (peak - new_portfolio_value) / peak
                raw_components['drawdown'] = current_drawdown * 100  # Convert to percentage

        # 4. Sentiment Alignment
        if hasattr(self, 'sentiment_scores') and action < self.n_stocks * 9:
            stock_idx = action // 9
            action_type = action % 9
            symbol = self.symbols[stock_idx]

            if symbol in self.sentiment_scores:
                try:
                    # Get current sentiment for this stock
                    if self.current_date in self.sentiment_scores[symbol].index:
                        current_sentiment = self.sentiment_scores[symbol].loc[self.current_date]

                        # Update sentiment_debug_info
                        sentiment_debug_info = {
                            'available': True,
                            'symbol': symbol,
                            'sentiment': float(current_sentiment),
                            'action_type': int(action_type)
                        }
                        # Calculate volatility adjustment factor
                        volatility_factor = 0.5  # Default value
                        try:
                            if self.current_date_idx > 20:
                                recent_prices = []
                                for i in range(max(0, self.current_date_idx-20), self.current_date_idx+1):
                                    if i < len(self.common_dates):
                                        date = self.common_dates[i]
                                        if date in self.stock_dfs[symbol].index:
                                            recent_prices.append(self.stock_dfs[symbol].loc[date]['Close'])

                                if len(recent_prices) > 5:
                                    volatility = np.std(recent_prices) / np.mean(recent_prices)
                                    volatility_factor = 1 / (1 + 5 * volatility)  # Reduce impact in volatile periods
                        except Exception as e:
                            print(f"Error calculating volatility for {symbol}: {e}")

                        # Determine if action aligns with sentiment
                        if action_type == 0:  # No action
                            alignment = 0  # Neutral for no action
                        elif (1 <= action_type <= 4 and current_sentiment > 0):  # Buy on positive
                            alignment = current_sentiment * volatility_factor
                        elif (5 <= action_type <= 8 and current_sentiment < 0):  # Sell on negative
                            alignment = -current_sentiment * volatility_factor
                        else:  # Misaligned action
                            alignment = -0.1 * volatility_factor  # Reduced penalty

                        raw_components['sentiment_alignment'] = alignment * 5  # Scale for normalization
                except (KeyError, TypeError) as e:
                    print(f"Error processing sentiment for {symbol}: {e}")

        # Normalize each component to [-1, 1] range
        normalized_components = {}

        # Define normalization parameters (based on typical ranges)
        norm_params = {
            'portfolio_change': {'min': -5, 'max': 5},     # Daily returns typically within ±5%
            'sharpe_ratio': {'min': -3, 'max': 3},         # Typical Sharpe ratio range
            'drawdown': {'min': 0, 'max': 20},             # Drawdowns from 0-20%
            'sentiment_alignment': {'min': -1, 'max': 1}   # Sentiment scores typically normalized
        }

        # Apply normalization to each component
        for component, value in raw_components.items():
            min_val = norm_params[component]['min']
            max_val = norm_params[component]['max']

            # Handle drawdown specially (convert to penalty)
            if component == 'drawdown':
                # Normalize and negate (higher drawdown = more negative reward)
                normalized_val = -np.clip((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)
            else:
                # Standard min-max normalization to [-1, 1]
                normalized_val = np.clip((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)

            normalized_components[component] = normalized_val

        # Calculate weighted normalized reward
        total_reward = (
            0.45 * normalized_components.get('portfolio_change', 0) +
            0.2 * normalized_components.get('sharpe_ratio', 0) +
            0.2 * normalized_components.get('drawdown', 0) +
            0.15 * normalized_components.get('sentiment_alignment', 0)
        )

        # Store both raw and normalized components for debugging
        reward_components = {
            'raw': {k: float(v) for k, v in raw_components.items()},
            'normalized': {k: float(v) for k, v in normalized_components.items()}
        }

        # Prepare info dictionary
        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "holdings": {s: self.portfolio.get(s, 0) for s in self.symbols},
            "date": self.current_date if not self.done and self.current_date else self.common_dates[-1] if self.common_dates else None,
            "reward_components": reward_components,
            "sentiment_debug_info": sentiment_debug_info
        }

        # Print reward components periodically
        if self.current_date_idx % 50 == 1:
            print(f"--- Step {self.current_date_idx} ({self.current_date}) ---")
            print(f"Raw Components: {', '.join([f'{k}: {v:.4f}' for k, v in raw_components.items()])}")
            print(f"Normalized: {', '.join([f'{k}: {v:.4f}' for k, v in normalized_components.items()])}")
            print(f"Total Reward: {total_reward:.4f}")
            print(f"Portfolio Value: {new_portfolio_value:.2f}")

            # Print sentiment info if available
            if sentiment_debug_info['available']:
                print(f"Symbol: {sentiment_debug_info['symbol']}, Date: {self.current_date}, "
                      f"Sentiment: {sentiment_debug_info['sentiment']}, Action: {sentiment_debug_info['action_type']}")
                if 'is_proxy' in sentiment_debug_info and sentiment_debug_info['is_proxy']:
                    print("(Using price momentum as sentiment proxy)")
            else:
                print("No sentiment alignment calculation performed for this step")

            print(f"-------------------------------------")

        return next_observation, total_reward, self.done, info


    # ===== END of Replacement =====

    def _calculate_stocks_value(self):
        """Calculate the current value of all stocks in the portfolio"""
        if not hasattr(self, 'current_date') or self.current_date is None:
            print("Warning: current_date not set in _calculate_stocks_value")
            return 0

        total_value = 0
        try:
            for symbol, shares in self.portfolio.items():
                if shares > 0:
                    price = self.stock_dfs[symbol].loc[self.current_date]['Close']
                    total_value += shares * price
        except KeyError as e:
            print(f"Error in _calculate_stocks_value: Date {self.current_date} not found. {e}")
            # Try using the most recent date available
            try:
                available_dates = self.stock_dfs[list(self.portfolio.keys())[0]].index
                closest_date = available_dates[available_dates <= pd.to_datetime(self.current_date)][-1]
                print(f"Using closest available date: {closest_date}")

                for symbol, shares in self.portfolio.items():
                    if shares > 0:
                        price = self.stock_dfs[symbol].loc[closest_date]['Close']
                        total_value += shares * price
            except Exception as nested_e:
                print(f"Fallback also failed: {nested_e}")
        except Exception as e:
            print(f"Unexpected error in _calculate_stocks_value: {e}")

        return total_value

    def _get_observation(self):
        stock_data = []
        for symbol in self.symbols:
            df = self.stock_dfs[symbol]
            window_dates = self.common_dates[self.current_date_idx - self.window_size : self.current_date_idx]
            window_data = df.loc[window_dates].values
            stock_data.append(window_data)

        combined_stock_data = np.hstack(stock_data)
        total_value = self.balance + self._calculate_stocks_value()
        allocations = []
        for symbol in self.symbols:
            shares = self.portfolio[symbol]
            price = self.stock_dfs[symbol].loc[self.current_date]['Close']
            stock_value = shares * price
            allocation = stock_value / total_value if total_value > 0 else 0
            allocations.append(allocation)

        portfolio_info = np.zeros((self.window_size, 2 + self.n_stocks))
        portfolio_info[:, 0] = self.balance / self.initial_balance  # Cash ratio
        portfolio_info[:, 1] = total_value / self.initial_balance  # Total value ratio
        portfolio_info[:, 2:] = allocations  # Stock allocations

        # Combine stock data with portfolio information
        observation = np.hstack((combined_stock_data, portfolio_info))

        return observation

    def _calculate_kelly_position(self):
        """Calculate position size using Kelly Criterion"""
        if self.win_rate == 0 or self.win_loss_ratio == 0:
            return 0.1  # Default to 10% position
        return self.win_rate - (1 - self.win_rate)/self.win_loss_ratio

    def _update_position_cost(self, symbol, price, shares):
        """Track average cost basis for each position"""
        if symbol not in self.position_cost:
            self.position_cost[symbol] = PositionTracker()

        if shares > 0:  # Buying
            self.position_cost[symbol].add_purchase(shares, price)
        else:  # Selling
            self.position_cost[symbol].reduce_position(-shares)

    def _take_action(self, action):
        """Execute trade action with position sizing and dynamic rewards"""
        # Initialize variables with default values
        reward = 0
        transaction_cost = 0
        symbol = None  # Initialize symbol for safety in error cases
        last_action_index = self.n_stocks * 9
        if int(action) == last_action_index:
            # This is the "+1" action, e.g., rebalance or hold all
            # Implement desired behavior here. For now, let's treat it as Hold All.
            print(f"Info: Action {action} interpreted as Hold All.")
            try:
                # Call the rebalance portfolio method and get the reward
                rebalance_reward = self._rebalance_portfolio()
                # Return the reward from rebalancing and 0 for transaction cost
                # (transaction costs are handled inside _rebalance_portfolio)
                return rebalance_reward, 0
            except Exception as e:
                print(f"Error during portfolio rebalancing: {e}")
                import traceback
                traceback.print_exc()
                return -5, 0
        try:
            # --- Action Interpretation and Symbol Determination FIRST ---
            action_type = int(action) % 9  # Ensure action is integer; 0-8 actions per stock
            stock_idx = int(action) // 9

            # Validate stock index before accessing self.symbols
            if not (0 <= stock_idx < len(self.symbols)):
                print(f"Warning: Invalid stock index {stock_idx} derived from action {action}. Taking no action.")
                return 0, 0 # Return neutral reward and zero cost

            symbol = self.symbols[stock_idx] # Assign symbol based on index
            # --- END Interpretation ---

            # --- Get Current Price AFTER symbol is known ---
            try:
                current_price = self.stock_dfs[symbol].loc[self.current_date]['Close']
            except KeyError:
                # ... (Error handling for price fetch remains the same) ...
                print(f"Warning: Date {self.current_date} not found for {symbol}. Using previous day's close or skipping action.")
                try:
                    current_date_dt = pd.to_datetime(self.current_date)
                    date_index = self.common_dates.index(current_date_dt)
                    prev_idx = date_index - 1
                    if prev_idx >= 0:
                        prev_date = self.common_dates[prev_idx]
                        current_price = self.stock_dfs[symbol].loc[prev_date]['Close']
                        print(f"Using price from {prev_date}: {current_price}")
                    else:
                        print(f"Cannot get previous price for {symbol} on {self.current_date}.")
                        return 0, 0
                except (ValueError, IndexError, KeyError, AttributeError) as fallback_e:
                    print(f"Fallback price retrieval failed for {symbol} on {self.current_date}: {fallback_e}")
                    return 0, 0

            # --- Now proceed with action logic ---
            portfolio_value = self.balance + self._calculate_stocks_value()
            max_pos_size_attr = getattr(self, 'max_position_size', 0.3)
            #kelly_fraction = min(self._calculate_kelly_position(), 0.5)

            # --- Handle Action Type ---
            if action_type == 0: # No action
                 # ... (Holding penalty/reward logic remains the same) ...
                 holding_reward_penalty = 0
                 if symbol in self.portfolio and self.portfolio[symbol] > 0 and symbol in self.position_cost:
                     avg_cost = self.position_cost[symbol].get_avg_cost()
                     if avg_cost > 0:
                         current_pnl_pct = (current_price - avg_cost) / avg_cost
                         if current_pnl_pct < -0.02:
                             holding_reward_penalty = current_pnl_pct * 10
                         elif current_pnl_pct > 0.01:
                             holding_reward_penalty = 0.1
                 return holding_reward_penalty, 0

            # Buy actions (1-4)
            elif 1 <= action_type <= 4:
                buy_percentage = action_type * 0.25

                # --- MODIFIED Risk Capital Calculation ---
                # Use a simpler approach: Risk a fixed fraction (e.g., 2-5%) of portfolio per trade, adjusted by buy_percentage
                base_risk_fraction = 0.05 # Example: Risk up to 5% of portfolio value on a 100% buy action
                potential_risk_capital = portfolio_value * base_risk_fraction
                # Ensure risk capital doesn't exceed half the portfolio or the available balance
                risk_capital = min(potential_risk_capital, portfolio_value * 0.5, self.balance)
                # --- END MODIFICATION ---

                amount_to_buy_value_step1 = min(risk_capital * buy_percentage, self.balance)

                current_stock_value = self.portfolio.get(symbol, 0) * current_price
                max_value_for_this_stock = portfolio_value * max_pos_size_attr
                allowed_additional_value = max(0, max_value_for_this_stock - current_stock_value)

                amount_to_buy_value = min(amount_to_buy_value_step1, allowed_additional_value)
                if amount_to_buy_value <= 1.0:
                    return -0.5 * buy_percentage, 0 # Return the penalty

                # ... (rest of the buy execution logic) ...
                shares_bought = amount_to_buy_value / current_price
                cost = amount_to_buy_value * (1 + self.transaction_fee_percent)

                if cost > self.balance:
                     return -0.5 * buy_percentage, 0

                self.balance -= cost
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + shares_bought
                transaction_cost = amount_to_buy_value * self.transaction_fee_percent
                self._update_position_cost(symbol, current_price, shares_bought)
                reward = 0.5 * buy_percentage

            # Sell actions (5-8)
            elif 5 <= action_type <= 8:
                # ... (Sell logic remains the same) ...
                sell_percentage = (action_type - 4) * 0.25
                shares_owned = self.portfolio.get(symbol, 0)
                shares_to_sell = shares_owned * sell_percentage

                if shares_to_sell <= 1e-6:
                    return -0.5 * sell_percentage, 0

                sale_value = shares_to_sell * current_price
                proceeds = sale_value * (1 - self.transaction_fee_percent)

                self.balance += proceeds
                self.portfolio[symbol] -= shares_to_sell
                transaction_cost = sale_value * self.transaction_fee_percent

                avg_cost = self.position_cost[symbol].get_avg_cost()
                if avg_cost > 0:
                    price_change_pct = (current_price - avg_cost) / avg_cost
                    reward = price_change_pct * sell_percentage * 100
                else:
                    reward = 0

                self._update_position_cost(symbol, current_price, -shares_to_sell)


            final_reward = reward
            final_reward = np.clip(final_reward, -25, 25)

            return final_reward, transaction_cost

        # ... (Error handling blocks remain the same) ...
        except IndexError as e:
            print(f"Error interpreting action {action} (IndexError): {e}. Symbol: {symbol}")
            return -5, 0
        except KeyError as e:
             print(f"Error accessing data for {symbol if symbol else 'Unknown Symbol'} on date {self.current_date} (KeyError): {e}")
             return -5, 0
        except Exception as e:
            symbol_str = symbol if symbol else "Unknown Symbol (error occurred before assignment)"
            print(f"Unexpected error executing action {action} for {symbol_str}: {e}")
            import traceback
            traceback.print_exc()
            return -10, 0

# ===== END of Modification =====


    def _rebalance_portfolio(self):
    # Calculate total portfolio value
      total_value = self.balance + self._calculate_stocks_value()

    # Target allocation per stock
      target_allocation = 1.0 / (self.n_stocks + 1)  # +1 for cash
      target_stock_value = total_value * target_allocation
      target_cash = target_stock_value

      reward = 0

    # First sell overweight positions
      for symbol in self.symbols:
        current_price = self.stock_dfs[symbol].loc[self.current_date]['Close']
        current_shares = self.portfolio[symbol]
        current_value = current_price * current_shares

        if current_value > target_stock_value:
            # Overweight position, need to sell
            value_to_sell = current_value - target_stock_value
            shares_to_sell = value_to_sell / current_price

            if shares_to_sell > 0:
                sell_value = shares_to_sell * current_price
                transaction_cost = sell_value * self.transaction_fee_percent

                # Execute sell
                self.balance += (sell_value - transaction_cost)
                self.portfolio[symbol] -= shares_to_sell

                self.trade_history[symbol].append({
                    'date': self.current_date,
                    'type': 'rebalance_sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'cost': transaction_cost,
                    'proceeds': sell_value - transaction_cost
                })

                reward += 0.5

    # Then buy underweight positions
      for symbol in self.symbols:
        current_price = self.stock_dfs[symbol].loc[self.current_date]['Close']
        current_shares = self.portfolio[symbol]
        current_value = current_price * current_shares

        if current_value < target_stock_value and self.balance > 0:
            # Underweight position, need to buy
            value_to_buy = min(target_stock_value - current_value, self.balance)

            if value_to_buy > 0:
                shares_to_buy = value_to_buy / current_price
                transaction_cost = value_to_buy * self.transaction_fee_percent

                if value_to_buy + transaction_cost > self.balance:
                    value_to_buy = self.balance / (1 + self.transaction_fee_percent)
                    shares_to_buy = value_to_buy / current_price
                    transaction_cost = value_to_buy * self.transaction_fee_percent

                # Execute buy
                self.balance -= (value_to_buy + transaction_cost)
                self.portfolio[symbol] += shares_to_buy

                self.trade_history[symbol].append({
                    'date': self.current_date,
                    'type': 'rebalance_buy',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': value_to_buy + transaction_cost
                })

                reward += 0.5

      return reward

    def render(self, mode='human'):
      if mode != 'human':
        return

    # Calculate portfolio value history
      portfolio_values = np.array(self.portfolio_value_history)

    # Calculate returns
      if len(portfolio_values) > 1:
        returns = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        print(f"Portfolio value: ${portfolio_values[-1]:.2f}")
        print(f"Total return: {returns:.2f}%")

        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()

        # Plot allocation over time if available
        if hasattr(self, 'allocation_history') and len(self.allocation_history) > 0:
            allocations = pd.DataFrame(self.allocation_history)

            plt.figure(figsize=(12, 6))
            allocations.plot(kind='area', figsize=(12, 6), stacked=True)
            plt.title('Portfolio Allocation Over Time')
            plt.xlabel('Trading Days')
            plt.ylabel('Allocation (%)')
            plt.grid(True)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MultiStockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_dfs, initial_balance=100000, transaction_fee_percent=0.0075, window_size=20):
        super().__init__()
        self.stock_dfs = stock_dfs
        self.symbols = list(stock_dfs.keys())
        self.n_stocks = len(self.symbols)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent

        # Compute common trading dates across all stock DataFrames
        common_dates = None
        for df in stock_dfs.values():
            common_dates = set(df.index) if common_dates is None else common_dates.intersection(df.index)
        self.common_dates = sorted(list(common_dates))
        print(f"Trading environment created with {len(self.common_dates)} common trading days")

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_stocks * 9 + 1)
        self.feature_count = len(next(iter(stock_dfs.values())).columns)
        portfolio_features = 2 + self.n_stocks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_stocks * self.feature_count + portfolio_features),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_date_idx = self.window_size
        self.current_date = self.common_dates[self.current_date_idx]
        self.balance = self.initial_balance
        self.portfolio = {symbol: 0 for symbol in self.symbols}
        self.portfolio_value_history = []
        self.done = False

        self.allocation_history = []
        self.trade_history = {symbol: [] for symbol in self.symbols}
        self.portfolio_value = self.balance + self._calculate_stocks_value()
        self.portfolio_value_history.append(self.portfolio_value)

        return self._get_observation()

    def _calculate_stocks_value(self):
        return sum(
            shares * self.stock_dfs[symbol].loc[self.current_date]['Close']
            for symbol, shares in self.portfolio.items()
        )

    def _get_observation(self):
        # Gather recent stock data for all symbols
        stock_data = []
        for symbol in self.symbols:
            df = self.stock_dfs[symbol]
            window_dates = self.common_dates[self.current_date_idx - self.window_size:self.current_date_idx]
            stock_data.append(df.loc[window_dates].values)

        combined_stock_data = np.hstack(stock_data)
        total_value = self.balance + self._calculate_stocks_value()

        # Prepare portfolio state info: balance, total value, allocations
        allocations = []
        for symbol in self.symbols:
            price = self.stock_dfs[symbol].loc[self.current_date]['Close']
            stock_value = self.portfolio[symbol] * price
            allocations.append(stock_value / total_value if total_value > 0 else 0)

        portfolio_info = np.zeros((self.window_size, 2 + self.n_stocks))
        portfolio_info[:, 0] = self.balance / self.initial_balance
        portfolio_info[:, 1] = total_value / self.initial_balance
        portfolio_info[:, 2:] = allocations

        return np.hstack((combined_stock_data, portfolio_info))

    def step(self, action):
        # Check if the episode is already marked as done
        if self.done:
            last_observation = self._get_observation() if hasattr(self, 'current_date') else np.zeros(self.observation_space.shape, dtype=np.float32)
            return last_observation, 0, True, {"portfolio_value": self.portfolio_value}

        sentiment_debug_info = {'available': False}
        prev_portfolio_value = self.portfolio_value
        total_transaction_cost = 0

        try:
            trade_reward, transaction_cost = self._take_action(action)
            total_transaction_cost += transaction_cost
        except Exception as e:
            trade_reward = -10
            transaction_cost = 0
            total_transaction_cost = 0

        self.current_date_idx += 1
        self.done = self.current_date_idx >= len(self.common_dates)

        new_portfolio_value = self.portfolio_value
        next_observation = None

        if not self.done:
            try:
                self.current_date = self.common_dates[self.current_date_idx]
            except IndexError:
                self.done = True
                self.current_date = self.common_dates[-1] if self.common_dates else None

            try:
                if self.current_date:
                    new_portfolio_value = self.balance + self._calculate_stocks_value()
                    self.portfolio_value = new_portfolio_value
                    self.portfolio_value_history.append(new_portfolio_value)
                else:
                    new_portfolio_value = prev_portfolio_value
            except Exception as e:
                new_portfolio_value = prev_portfolio_value

            try:
                if self.current_date:
                    next_observation = self._get_observation()
                else:
                    next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                    self.done = True
            except Exception as e:
                next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                self.done = True
        else:
            new_portfolio_value = self.portfolio_value
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ================== Enhanced Normalized Reward Calculation ==================
        raw_components = {
            'portfolio_change': 0,
            'sharpe_ratio': 0,
            'drawdown': 0,
            'sentiment_alignment': 0
        }

        if prev_portfolio_value > 0:
            portfolio_change = (new_portfolio_value / prev_portfolio_value) - 1
            raw_components['portfolio_change'] = portfolio_change * 100

        if len(self.portfolio_value_history) > 30:
            returns = np.diff(np.log(self.portfolio_value_history[-30:]))
            if len(returns) > 1 and np.std(returns) > 1e-9:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                raw_components['sharpe_ratio'] = sharpe

        if len(self.portfolio_value_history) > 1:
            peak = np.max(self.portfolio_value_history)
            if peak > 0:
                current_drawdown = (peak - new_portfolio_value) / peak
                raw_components['drawdown'] = current_drawdown * 100

        if hasattr(self, 'sentiment_scores') and action < self.n_stocks * 9:
            stock_idx = action // 9
            action_type = action % 9
            symbol = self.symbols[stock_idx]
            if symbol in self.sentiment_scores:
                try:
                    if self.current_date in self.sentiment_scores[symbol].index:
                        current_sentiment = self.sentiment_scores[symbol].loc[self.current_date]
                        sentiment_debug_info = {
                            'available': True,
                            'symbol': symbol,
                            'sentiment': float(current_sentiment),
                            'action_type': int(action_type)
                        }
                        volatility_factor = 0.5
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
                                    volatility_factor = 1 / (1 + 5 * volatility)
                        except Exception as e:
                            pass
                        if action_type == 0:
                            alignment = 0
                        elif (1 <= action_type <= 4 and current_sentiment > 0):
                            alignment = current_sentiment * volatility_factor
                        elif (5 <= action_type <= 8 and current_sentiment < 0):
                            alignment = -current_sentiment * volatility_factor
                        else:
                            alignment = -0.1 * volatility_factor
                        raw_components['sentiment_alignment'] = alignment * 5
                except (KeyError, TypeError):
                    pass

        normalized_components = {}
        norm_params = {
            'portfolio_change': {'min': -5, 'max': 5},
            'sharpe_ratio': {'min': -3, 'max': 3},
            'drawdown': {'min': 0, 'max': 20},
            'sentiment_alignment': {'min': -1, 'max': 1}
        }

        for component, value in raw_components.items():
            min_val = norm_params[component]['min']
            max_val = norm_params[component]['max']
            if component == 'drawdown':
                normalized_val = -np.clip((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)
            else:
                normalized_val = np.clip((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)
            normalized_components[component] = normalized_val

        total_reward = (
            0.45 * normalized_components.get('portfolio_change', 0) +
            0.2 * normalized_components.get('sharpe_ratio', 0) +
            0.2 * normalized_components.get('drawdown', 0) +
            0.15 * normalized_components.get('sentiment_alignment', 0)
        )

        reward_components = {
            'raw': {k: float(v) for k, v in raw_components.items()},
            'normalized': {k: float(v) for k, v in normalized_components.items()}
        }

        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "holdings": {s: self.portfolio.get(s, 0) for s in self.symbols},
            "date": self.current_date if not self.done and self.current_date else self.common_dates[-1] if self.common_dates else None,
            "reward_components": reward_components,
            "sentiment_debug_info": sentiment_debug_info
        }

        return next_observation, total_reward, self.done, info

    def _take_action(self, action):
        prev_value = self.portfolio_value
        if action == self.n_stocks * 9:
            return self._rebalance_portfolio()

        stock_idx, action_type = divmod(action, 9)
        if stock_idx >= self.n_stocks:
            return 0

        symbol = self.symbols[stock_idx]
        price = self.stock_dfs[symbol].loc[self.current_date]['Close']

        if action_type == 0:
            return 0  
        elif 1 <= action_type <= 4:
            buy_pct = action_type * 0.25
            amount = self.balance * buy_pct
            if amount < price:
                return 0
            shares = int(amount / (price * (1 + self.transaction_fee_percent)))
            cost = shares * price * (1 + self.transaction_fee_percent)
            if shares > 0:
                self.balance -= cost
                self.portfolio[symbol] += shares
                self.trade_history[symbol].append(('BUY', self.current_date, shares, price))
        elif 5 <= action_type <= 8:
            sell_pct = (action_type - 4) * 0.25
            shares_to_sell = int(self.portfolio[symbol] * sell_pct)
            if shares_to_sell <= 0:
                return 0
            proceeds = shares_to_sell * price * (1 - self.transaction_fee_percent)
            self.balance += proceeds
            self.portfolio[symbol] -= shares_to_sell
            self.trade_history[symbol].append(('SELL', self.current_date, shares_to_sell, price))

        new_value = self.balance + self._calculate_stocks_value()
        return (new_value - prev_value) / prev_value

    def _rebalance_portfolio(self):
        prev_value = self.portfolio_value

        # Sell all current holdings
        for symbol, shares in self.portfolio.items():
            if shares > 0:
                price = self.stock_dfs[symbol].loc[self.current_date]['Close']
                proceeds = shares * price * (1 - self.transaction_fee_percent)
                self.balance += proceeds
                self.portfolio[symbol] = 0
                self.trade_history[symbol].append(('SELL', self.current_date, shares, price))

        # Buy equal-value allocation for each stock
        alloc_per_stock = self.balance / self.n_stocks
        for symbol in self.symbols:
            price = self.stock_dfs[symbol].loc[self.current_date]['Close']
            shares = int(alloc_per_stock / (price * (1 + self.transaction_fee_percent)))
            if shares > 0:
                cost = shares * price * (1 + self.transaction_fee_percent)
                self.balance -= cost
                self.portfolio[symbol] += shares
                self.trade_history[symbol].append(('BUY', self.current_date, shares, price))

        new_value = self.balance + self._calculate_stocks_value()
        return (new_value - prev_value) / prev_value

    def render(self, mode='human'):
        if mode != 'human':
            return

        fig = plt.figure(figsize=(20, 12))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_title('Portfolio Value Over Time')
        ax1.plot(self.common_dates[self.window_size:self.current_date_idx+1], self.portfolio_value_history)
        ax1.set_ylabel('Portfolio Value (₹)')
        
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.set_title('Stock Prices')
        for symbol in self.symbols:
            prices = self.stock_dfs[symbol].loc[
                self.common_dates[self.window_size:self.current_date_idx+1]
            ]['Close']
            ax2.plot(prices.index, prices / prices.iloc[0], label=symbol)
        ax2.set_ylabel('Normalized Price')
        ax2.legend()
        
        if self.allocation_history:
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.set_title('Portfolio Allocation')
            alloc_matrix = np.zeros((len(self.allocation_history), len(self.symbols)))
            for i, alloc in enumerate(self.allocation_history):
                for j, symbol in enumerate(self.symbols):
                    alloc_matrix[i, j] = alloc.get(symbol, 0)
            cash_alloc = 1 - np.sum(alloc_matrix, axis=1)
            dates = self.common_dates[self.window_size+1:self.current_date_idx+1]
            stacked = np.vstack([alloc_matrix.T, cash_alloc])
            ax3.stackplot(dates, stacked, labels=self.symbols + ['Cash'], alpha=0.7)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Allocation %')
            ax3.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
import numpy as np
from risk.risk_manager import RiskManager


class RiskManager:
    def __init__(self, max_position_size=0.2, stop_loss_pct=0.05,
                 max_drawdown=0.15, volatility_lookback=20):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback

    def calculate_position_size(self, portfolio_value, price_history, symbol):
        # Use ATR-based volatility for dynamic sizing
        if len(price_history) < self.volatility_lookback:
            return self.max_position_size
        
        highs = price_history['High'].values[-self.volatility_lookback:]
        lows = price_history['Low'].values[-self.volatility_lookback:]
        closes = price_history['Close'].values[-self.volatility_lookback:-1]
        
        tr1 = highs - lows
        tr2 = np.abs(highs - np.append([closes[0]], closes))
        tr3 = np.abs(lows - np.append([closes[0]], closes))
        
        true_ranges = np.vstack([tr1, tr2, tr3]).max(axis=0)
        atr = true_ranges.mean()
        current_price = price_history['Close'].values[-1]
        volatility = atr / current_price

        position_size = self.max_position_size * (0.05 / max(volatility, 0.005))
        return min(position_size, self.max_position_size)

    def check_stop_loss(self, entry_price, current_price, position_type='long'):
        # Exit if price crosses stop-loss threshold
        if position_type == 'long':
            return current_price < entry_price * (1 - self.stop_loss_pct)
        else:
            return current_price > entry_price * (1 + self.stop_loss_pct)

    def check_drawdown(self, peak_value, current_value):
        if peak_value == 0:
            return False
        drawdown = (peak_value - current_value) / peak_value
        return drawdown > self.max_drawdown

    def kelly_position_size(self, win_rate, win_loss_ratio, max_risk=0.02):
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        return min(kelly * 0.5, max_risk)

    def integrate_with_trading_env(self, env, stock_dfs):
        # Wrap and enhance the trading environment with risk logic
        class RiskManagedTradingEnv(env.__class__):
            def __init__(self_env, *args, **kwargs):
                super(RiskManagedTradingEnv, self_env).__init__(*args, **kwargs)
                self_env.risk_manager = self
                self_env.position_entry_prices = {symbol: 0 for symbol in self_env.symbols}
                self_env.peak_portfolio_value = self_env.initial_balance
                self_env.trailing_win_rate = 0.5
                self_env.win_count = 0
                self_env.loss_count = 0
                self_env.win_amount = 0
                self_env.loss_amount = 0

            def _take_action(self_env, action):
                prev_portfolio_value = self_env.portfolio_value
                for symbol, shares in self_env.portfolio.items():
                    if shares > 0:
                        entry_price = self_env.position_entry_prices[symbol]
                        current_price = self_env.stock_dfs[symbol].loc[self_env.current_date]['Close']
                        if entry_price > 0 and self.check_stop_loss(entry_price, current_price):
                            proceeds = shares * current_price * (1 - self_env.transaction_fee_percent)
                            self_env.balance += proceeds
                            self_env.portfolio[symbol] = 0
                            self_env.trade_history[symbol].append(
                                ('STOP_LOSS', self_env.current_date, shares, current_price)
                            )

                current_value = self_env.balance + self_env._calculate_stocks_value()
                if current_value > self_env.peak_portfolio_value:
                    self_env.peak_portfolio_value = current_value

                if self.check_drawdown(self_env.peak_portfolio_value, current_value):
                    for symbol, shares in self_env.portfolio.items():
                        if shares > 0:
                            current_price = self_env.stock_dfs[symbol].loc[self_env.current_date]['Close']
                            proceeds = shares * current_price * (1 - self_env.transaction_fee_percent)
                            self_env.balance += proceeds
                            self_env.portfolio[symbol] = 0
                            self_env.trade_history[symbol].append(
                                ('MAX_DRAWDOWN', self_env.current_date, shares, current_price)
                            )
                    return -0.1  

                result = super(RiskManagedTradingEnv, self_env)._take_action(action)
                new_value = self_env.balance + self_env._calculate_stocks_value()
                if new_value != prev_portfolio_value:
                    if new_value > prev_portfolio_value:
                        self_env.win_count += 1
                        self_env.win_amount += (new_value - prev_portfolio_value)
                    else:
                        self_env.loss_count += 1
                        self_env.loss_amount += (prev_portfolio_value - new_value)

                    trade_count = self_env.win_count + self_env.loss_count
                    if trade_count > 0:
                        self_env.trailing_win_rate = self_env.win_count / trade_count
                        avg_win = self_env.win_amount / max(self_env.win_count, 1)
                        avg_loss = self_env.loss_amount / max(self_env.loss_count, 1)
                        win_loss_ratio = avg_win / max(avg_loss, 1)
                        self_env.kelly_factor = self.kelly_position_size(
                            self_env.trailing_win_rate, win_loss_ratio
                        )

                return result

            def get_position_size(self_env, symbol, portfolio_value):
                price_history = self_env.stock_dfs[symbol].loc[:self_env.current_date].tail(30)
                base_size = self.calculate_position_size(portfolio_value, price_history, symbol)
                if hasattr(self_env, 'kelly_factor'):
                    return min(base_size, self_env.kelly_factor)
                return base_size

        risk_env = RiskManagedTradingEnv(stock_dfs=stock_dfs)
        return risk_env

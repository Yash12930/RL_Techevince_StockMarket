# RL Equity Trading Framework

**RL\_StockMarket** is an advanced toolkit and research framework for applying Reinforcement Learning (RL) to stock portfolio management and trading strategies. It brings together state-of-the-art RL algorithms, technical indicators, sentiment analysis, and risk management in a modular pipeline for Indian and global stock markets.

---

## 📂 File Structure

```
project/
├── fetchinfunc.py        # Data pipeline (OHLCV + Technical Indicators)
├── Agent.py              # Core RL agent configuration (action/state/reward)
├── Hyperpar.py           # Hyperparameter optimization grid
├── A2cAgent.py           # A2C implementation (CNN-LSTM Policy)
├── Sentiment.py          # News scraping & VADER/FinBERT analysis
├── RiskManagement.py     # Position sizing & stop-loss logic
├── TradingEnv.py         # Custom Gym environment
├── Train.py              # Main Training and saving the model
└── backtester.py         # Backtesting the results
```

---

## 🔧 Key Features

### 🏛 Core Architecture

| Component       | Specification                           |
| --------------- | --------------------------------------- |
| RL Algorithm    | Advantage Actor-Critic (A2C)            |
| Neural Network  | CNN-LSTM Hybrid                         |
| Lookback Window | 20 trading days                         |
| Action Space    | 9 actions per stock (25%-100% buy/sell) |

---

### 📊 Data Pipeline

```python
{
  "Data Sources": ["Yahoo Finance (yfinance)"],
  "Stocks": ["RELIANCE.NS", "TCS.NS", ...],  # 10 NSE stocks
  "Features": [
    "OHLCV",
    "SMA20", "SMA50", 
    "EMA20",
    "RSI",
    "MACD",
  ],
  "Normalization": "Z-score"
}
```

---

### 🧠 Sentiment Engine

```python
sentiment_config = {
  "Sources": ["Yahoo Finance News"],
  "Scraping": "BeautifulSoup",
  "Models": ["VADER", "FinBERT"],
  "Temporal": "3-day rolling window",
  "Weight": 0.15  # Reward component weight
}
```

---

### ⚖️ Risk Management

| Mechanism       | Threshold           | Implementation     |
| --------------- | ------------------- | ------------------ |
| Stop-Loss       | 5% trailing         | Dynamic adjustment |
| Max Drawdown    | 15% circuit breaker | Portfolio freeze   |

---

## 🎯 Reward Function

The total reward for the RL agent is computed as a weighted combination of performance and sentiment metrics:

```python
total_reward = (
    0.45 * portfolio_change_norm +
    0.20 * sharpe_ratio_norm +
    0.20 * drawdown_norm +
    0.15 * sentiment_norm
)
```

This formulation encourages balanced trading decisions considering returns, risk, and market sentiment.

---

## 🚀 Installation & Usage

### Prerequisites

* Python 3.7+

### Example Usage

Edit `FinalCall.py` to specify stocks and training/testing periods:

```python
stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]
training_period = ("2018-01-01", "2022-01-01")
testing_period = ("2022-01-02", "2022-12-31")
```

Run the trading system:

```bash
git clone https://github.com/Yash12930/RL_Techevince_StockMarket.git
cd RL_Techevince_StockMarket
pip install -r requirements.txt
python backtester.py
```

---

## 📈 Performance Metrics

| Metric                  | Value           |
|------------------------|-----------------|
| Initial Portfolio Value| ₹10,00,000.00   |
| Final Portfolio Value  | ₹10,59,211.27   |
| Total Return           | 5.92%           |
| Maximum Drawdown       | 7.84%           |
| Sharpe Ratio (Annual)  | 0.6880          |

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss major changes or new features.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

* [yfinance](https://github.com/ranaroussi/yfinance)
* [TensorFlow](https://www.tensorflow.org/)
* [NLTK](https://www.nltk.org/)
* [Transformers (HuggingFace)](https://huggingface.co/)

---

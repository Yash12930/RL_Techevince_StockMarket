# RL_Techevince_StockMarket

RL_Techevince_StockMarket is an advanced toolkit and research framework for applying Reinforcement Learning (RL) to stock portfolio management and trading strategies. It brings together state-of-the-art RL algorithms, technical analysis, sentiment analysis, and risk management in a modular pipeline for Indian and global stock markets.

## Features

- **Data Fetching & Technical Analysis**
  - Automated retrieval of historical stock data using `yfinance`.
  - Calculation of technical indicators such as SMA, EMA, RSI, and MACD.
  - Data preprocessing with normalization and feature engineering for machine learning tasks.

- **Reinforcement Learning Agents**
  - Implementation of the Advantage Actor-Critic (A2C) agent with deep neural networks (CNN + LSTM).
  - Modular support for other RL algorithms (e.g., PPO).
  - Training and evaluation workflows for portfolio management.

- **Sentiment Analysis**
  - Integration of financial news sentiment using VADER and optional FinBERT transformer-based models.
  - Text cleaning and NLP pipelines for extracting sentiment signals from financial headlines and news.

- **Risk Management**
  - Tools for risk management and environment wrappers for safe trading simulation.

- **Performance Evaluation**
  - Comparison with Buy & Hold strategies.
  - ROI, profit/loss, and outperformance reporting.

## Quick Start

### Prerequisites

- Python 3.7+
- Install required libraries:

```bash
pip install -r requirements.txt
```

### Example Usage

Edit `FinalCall.py` to specify stocks, training period, and testing period:

```python
stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]
training_period = ("2018-01-01", "2022-01-01")
testing_period = ("2022-01-02", "2022-12-31")
```

Run the trading system:

```bash
python FinalCall.py
```

## File Structure

- `fetchinfunc.py`: Fetches stock data and computes technical indicators.
- `A2cAgent.py`: Contains the A2C agent implementation and training logic.
- `Sentiment.py`: News sentiment analysis tools and sentiment-enhanced trading environment.
- `RiskManagement.py`: (Assumed) Risk management utilities.
- `TradingEnv.py`: (Assumed) Custom OpenAI Gym environment for stock trading.
- `FinalCall.py`: Main pipeline for running the end-to-end RL trading system.

## Key Concepts

- **Technical Indicators**: SMA, EMA, RSI, MACD
- **Sentiment Analysis**: VADER, FinBERT (optional)
- **Reinforcement Learning**: Advantage Actor-Critic (A2C), PPO (plug-in ready)
- **Risk Management**: Adjustable modules for safe trading

## Results

The system outputs portfolio performance and compares RL strategies with traditional Buy & Hold, reporting ROI and profit/loss.

## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss any major changes.

## License

[MIT](LICENSE)

## Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance)
- [TensorFlow](https://www.tensorflow.org/)
- [NLTK](https://www.nltk.org/)
- [Transformers (HuggingFace)](https://huggingface.co/)
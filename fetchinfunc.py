# technical_analysis.py
"""
Comprehensive Financial Data Analysis Pipeline

A complete toolkit for fetching stock data, calculating technical indicators,
and preprocessing data for machine learning applications.
"""

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        if stock.empty:
            print(f"Warning: No data found for {ticker}")
            return None

        # Standardize date format and sort chronologicaly
        stock.index = pd.to_datetime(stock.index).strftime('%Y-%m-%d')
        stock = stock.sort_index()
        
        return stock
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def add_technical_indicators(df):
    df = df.copy()

    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculations
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Cleanup and return
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    df = df.copy()
    df.dropna(inplace=True)

    # Identify numerical features for scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if 'Close' in numeric_cols:
        numeric_cols = numeric_cols.drop('Close')  # Preserve original prices

    # Feature scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df


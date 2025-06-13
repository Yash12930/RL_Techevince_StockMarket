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
    """
    Fetch historical stock data using yfinance and handle MultiIndex structure
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"No data found for {ticker}.")
            return None

        # Ensure index is datetime
        stock_data.index = pd.to_datetime(stock_data.index)

        # Check if we have a MultiIndex structure
        if isinstance(stock_data.columns, pd.MultiIndex):
            print(f"MultiIndex detected for {ticker}, flattening...")

            # Create a new dataframe with flattened columns
            flattened_df = pd.DataFrame(index=stock_data.index)

            # Extract each column we need
            if ('Close', ticker) in stock_data.columns:
                flattened_df['Close'] = stock_data[('Close', ticker)]
                flattened_df['Open'] = stock_data[('Open', ticker)]
                flattened_df['High'] = stock_data[('High', ticker)]
                flattened_df['Low'] = stock_data[('Low', ticker)]
                flattened_df['Volume'] = stock_data[('Volume', ticker)]
            else:
                # Try to find the columns regardless of their exact structure
                for col in stock_data.columns:
                    if col[0] == 'Close':
                        flattened_df['Close'] = stock_data[col]
                    elif col[0] == 'Open':
                        flattened_df['Open'] = stock_data[col]
                    elif col[0] == 'High':
                        flattened_df['High'] = stock_data[col]
                    elif col[0] == 'Low':
                        flattened_df['Low'] = stock_data[col]
                    elif col[0] == 'Volume':
                        flattened_df['Volume'] = stock_data[col]

            return flattened_df
        else:
            # If not MultiIndex, return as is
            return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_simple_indicators(df):
    """
    Add a simplified set of technical indicators to reduce potential errors
    """
    if df is None or df.empty:
        return None

    result = df.copy()

    try:
        # Check if 'Close' is a DataFrame with multiple columns
        if isinstance(result['Close'], pd.DataFrame):
            print(f"Close is a DataFrame with columns: {result['Close'].columns}")
            # Extract just the 'Close' column as a Series
            close_series = result['Close']['Close'] if 'Close' in result['Close'].columns else result['Close'].iloc[:, 0]

            # Simple moving averages - directly using pandas
            result['SMA_20'] = close_series.rolling(window=20).mean()
            result['SMA_50'] = close_series.rolling(window=50).mean()

            # Relative price to moving averages
            result['Price_to_SMA20'] = close_series / result['SMA_20']
            result['Price_to_SMA50'] = close_series / result['SMA_50']

            # Daily returns
            result['Daily_Return'] = close_series.pct_change()

            # Volatility (20-day standard deviation of returns)
            result['Volatility'] = result['Daily_Return'].rolling(window=20).std()
        else:
            # Original approach for when 'Close' is a Series
            result['SMA_20'] = result['Close'].rolling(window=20).mean()
            result['SMA_50'] = result['Close'].rolling(window=50).mean()

            result['Price_to_SMA20'] = result['Close'] / result['SMA_20']
            result['Price_to_SMA50'] = result['Close'] / result['SMA_50']

            result['Daily_Return'] = result['Close'].pct_change()
            result['Volatility'] = result['Daily_Return'].rolling(window=20).std()

        # Fill missing values
        result.fillna(method='bfill', inplace=True)

        return result
    except Exception as e:
        print(f"Error adding indicators: {e}")
        print(f"Data structure: Close type = {type(result['Close'])}")
        if hasattr(result['Close'], 'shape'):
            print(f"Close shape = {result['Close'].shape}")
        if isinstance(result['Close'], pd.DataFrame):
            print(f"Close columns = {result['Close'].columns}")
        return None
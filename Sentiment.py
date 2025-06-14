import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime as dt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed. Internet connection required for first-time setup.")

class NewsSentimentAnalyzer:
    """
    Analyze news sentiment for stock market prediction
    """
    def __init__(self, use_transformers=True):
        self.use_transformers = use_transformers

        # Initialize VADER sentiment analyzer as fallback
        self.vader = SentimentIntensityAnalyzer()

        if use_transformers:
            try:
                print("Loading FinBERT model for sentiment analysis...")
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                print("FinBERT model loaded successfully.")
            except Exception as e:
                print(f"Failed to load FinBERT model: {e}. Falling back to VADER.")
                self.use_transformers = False

        # Initialize news cache to avoid repeated API calls
        self.news_cache = {}

    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_sentiment_vader(self, text):
        """Get sentiment score using VADER"""
        sentiment = self.vader.polarity_scores(text)
        return sentiment['compound']  # Range: -1 (negative) to 1 (positive)

    def get_sentiment_finbert(self, text):
        """Get sentiment score using FinBERT"""
        try:
            # Ensure text is not too long for the model
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():  # No need to track gradients for inference
                outputs = self.model(**inputs)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT classes: negative (0), neutral (1), positive (2)
            negative_score = probabilities[0][0].item()
            neutral_score = probabilities[0][1].item()
            positive_score = probabilities[0][2].item()

            # Normalize to [-1, 1] range
            score = positive_score - negative_score

            return score
        except Exception as e:
            print(f"FinBERT error: {e}")
            return self.get_sentiment_vader(text)  # Fallback to VADER

    def get_sentiment(self, text):
        """Get sentiment score using selected method"""
        text = self.clean_text(text)
        if self.use_transformers:
            return self.get_sentiment_finbert(text)
        else:
            return self.get_sentiment_vader(text)

    def fetch_news_yahoo(self, ticker, days_back=3):
        """Fetch news from Yahoo Finance with caching"""
        # Check cache first
        cache_key = f"{ticker}_{days_back}_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]

        news_data = []
        try:
            # Yahoo Finance URL
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract news items
            news_items = soup.find_all('div', {'class': 'Ov(h) Pend(44px) Pstart(25px)'})

            for item in news_items:
                try:
                    headline = item.find('h3').text
                    timestamp = item.find('span', {'class': 'C(#959595)'}).text

                    # Simple date parsing
                    date = dt.datetime.now().date()
                    if 'hours' in timestamp or 'minutes' in timestamp:
                        pass  # Today
                    elif 'yesterday' in timestamp.lower():
                        date = date - dt.timedelta(days=1)
                    elif 'days' in timestamp:
                        days = int(re.search(r'(\d+)', timestamp).group(1))
                        date = date - dt.timedelta(days=days)

                    if (dt.datetime.now().date() - date).days <= days_back:
                        news_data.append({'headline': headline, 'date': date})
                except Exception as e:
                    print(f"Error parsing news item: {e}")

            # Cache the results
            self.news_cache[cache_key] = news_data

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")

        return news_data

    def get_stock_sentiment(self, ticker, days_back=3):
        """Get overall sentiment for a stock"""
        news = self.fetch_news_yahoo(ticker, days_back)

        if not news:
            return 0  # Neutral if no news found

        # Calculate sentiment for each headline
        sentiments = []
        for item in news:
            sentiment = self.get_sentiment(item['headline'])
            sentiments.append(sentiment)

        # Return average sentiment if available, otherwise neutral (0)
        return sum(sentiments) / len(sentiments) if sentiments else 0

class SentimentEnhancedTradingEnv:
    @staticmethod
    def enhance_environment(base_env_class):
        class EnhancedEnv(base_env_class):
            def __init__(self, *args, **kwargs):
                # Initialize sentiment analyzer
              try:
                self.sentiment_analyzer = NewsSentimentAnalyzer(use_transformers=True)
                self.sentiment_scores = {}
                self.symbols = kwargs.get('stock_dfs', {}).keys()

                # Fetch and calculate sentiment for each stock
                print("Calculating sentiment scores for each stock...")
                for symbol in self.symbols:
                    stock_df = kwargs['stock_dfs'][symbol]
                    dates = stock_df.index

                    # Initialize with neutral sentiment
                    scores = np.zeros(len(dates))

                    # Calculate sentiment for each date with a sliding window
                    window_size = 7  # Look back 7 days for news
                    for i, date in enumerate(dates):
                        if i % 50 == 0:  # Progress indicator
                            print(f"Processing sentiment for {symbol}: {i}/{len(dates)}")

                        # For dates near the beginning, use a smaller window
                        lookback = min(i, window_size)

                        # Convert date to string format for sentiment analysis
                        date_str = date.strftime('%Y-%m-%d')

                        # Get sentiment for this date range
                        try:
                            # Get news for the past few days
                            news_items = self.sentiment_analyzer.fetch_news_yahoo(symbol, days_back=lookback)

                            # If no news, keep neutral sentiment
                            if not news_items:
                                scores[i] = 0
                                continue

                            # Calculate sentiment for each news item
                            sentiments = []
                            for item in news_items:
                                sentiment = self.sentiment_analyzer.get_sentiment(item['headline'])
                                sentiments.append(sentiment)

                            # Average sentiment for this date
                            if sentiments:
                                scores[i] = sum(sentiments) / len(sentiments)
                        except Exception as e:
                            print(f"Error calculating sentiment for {symbol} on {date_str}: {e}")
                            # Keep neutral sentiment on error
                            scores[i] = 0

                    # Create Series with calculated sentiment scores
                    self.sentiment_scores[symbol] = pd.Series(scores, index=dates)

                    # Smooth sentiment scores with exponential moving average
                    self.sentiment_scores[symbol] = self.sentiment_scores[symbol].ewm(span=5).mean()

                # Now call the parent init
                super(EnhancedEnv, self).__init__(*args, **kwargs)

                # Update observation space to include sentiment features
                if hasattr(self, 'observation_space') and isinstance(self.observation_space, spaces.Box):
                    base_shape = self.observation_space.shape
                    sentiment_feature_count = self.n_stocks
                    new_feature_dim = base_shape[1] + sentiment_feature_count

                    self.observation_space = spaces.Box(
                        low=self.observation_space.low.min(),
                        high=self.observation_space.high.max(),
                        shape=(base_shape[0], new_feature_dim),
                        dtype=np.float32
                    )
                    print(f"Sentiment Enhanced Env Observation Space Updated To: {self.observation_space.shape}")
                else:
                    print("Warning: Base observation_space not found or not Box. Attempting manual setup.")
                    window_size = kwargs.get('window_size', 20)
                    feature_count = 11
                    portfolio_features = 2 + self.n_stocks
                    sentiment_feature_count = self.n_stocks
                    total_features = feature_count * self.n_stocks + portfolio_features + sentiment_feature_count

                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(window_size, total_features),
                        dtype=np.float32
                    )
                    print(f"Sentiment Enhanced Env Observation Space MANUALLY Set To: {self.observation_space.shape}")
              except Exception as e:
                print(f"Error fetching news for {symbol}, using simulated sentiment: {e}")
                # Use simulated sentiment when real data fails
                if i == 0:  # Only generate once per stock
                    simulated_sentiments = generate_simulated_sentiment(dates, seed=hash(symbol) % 10000)
                scores[i] = simulated_sentiments[i]

            def generate_simulated_sentiment(dates, seed=None):
                """Generate simulated sentiment data when real news fetching fails"""
                if seed is not None:
                    np.random.seed(seed)

                # Create a base sentiment trend (slightly positive bias for stocks)
                base_trend = np.random.normal(0.05, 0.2, len(dates))

                # Add some autocorrelation (sentiment tends to persist)
                for i in range(1, len(base_trend)):
                    base_trend[i] = 0.8 * base_trend[i-1] + 0.2 * base_trend[i]

                # Add occasional sentiment shocks (news events)
                num_shocks = len(dates) // 30  # Approximately one shock per month
                shock_indices = np.random.choice(range(len(dates)), size=num_shocks, replace=False)
                shock_magnitudes = np.random.normal(0, 0.5, num_shocks)

                for idx, magnitude in zip(shock_indices, shock_magnitudes):
                    # Shock affects sentiment for next 3-7 days, gradually diminishing
                    shock_duration = np.random.randint(3, 8)
                    for j in range(shock_duration):
                        if idx + j < len(base_trend):
                            decay_factor = (shock_duration - j) / shock_duration
                            base_trend[idx + j] += magnitude * decay_factor

                # Clip to reasonable sentiment range [-1, 1]
                return np.clip(base_trend, -1, 1)

            def _get_observation(self):
                """Override to include sentiment data in the observation"""
                base_observation = super()._get_observation()

                # Add sentiment data for the current window
                sentiment_features = []
                for symbol in self.symbols:
                    sentiment = self.sentiment_scores[symbol]
                    # Extract sentiment for dates in the current window
                    start_idx = max(0, self.current_date_idx - self.window_size)
                    end_idx = self.current_date_idx

                    if hasattr(self, 'common_dates') and len(self.common_dates) > end_idx:
                        window_dates = self.common_dates[start_idx:end_idx]
                        # Filter the sentiment Series for these dates
                        window_sentiment = sentiment.loc[sentiment.index.isin(window_dates)].values
                    else:
                        # Fallback if common_dates isn't available
                        window_sentiment = sentiment.iloc[start_idx:end_idx].values

                    # Reshape to correct dimensions and pad if needed
                    if len(window_sentiment) < self.window_size:
                        padding = np.zeros(self.window_size - len(window_sentiment))
                        window_sentiment = np.append(padding, window_sentiment)

                    # Ensure 2D shape for hstack
                    window_sentiment = window_sentiment.reshape(-1, 1)
                    sentiment_features.append(window_sentiment)

                # Combine all sentiment data
                if sentiment_features:
                    combined_sentiment = np.hstack(sentiment_features)

                    # Add sentiment to observation
                    enhanced_observation = np.hstack((base_observation, combined_sentiment))
                else:
                    enhanced_observation = base_observation

                # Normalize the observation
                if enhanced_observation.ndim == 2:
                    mean = np.mean(enhanced_observation, axis=0, keepdims=True)
                    std = np.std(enhanced_observation, axis=0, keepdims=True)
                    enhanced_observation = (enhanced_observation - mean) / (std + 1e-8)
                    enhanced_observation = np.nan_to_num(enhanced_observation)

                return enhanced_observation

        return EnhancedEnv

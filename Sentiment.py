import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime as dt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed. Internet connection required for first-time setup.")

class NewsSentimentAnalyzer:
    def __init__(self, use_transformers=False):
        self.use_transformers = use_transformers

        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

        # Initialize FinBERT (if using transformers)
        if use_transformers:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            except:
                print("Failed to load FinBERT model. Falling back to VADER.")
                self.use_transformers = False

    def clean_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_sentiment_vader(self, text):
        sentiment = self.vader.polarity_scores(text)
        return sentiment['compound'] 

    def get_sentiment_finbert(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            negative_score = probabilities[0][0].item()
            positive_score = probabilities[0][2].item()
            score = positive_score - negative_score
            return score
        except Exception as e:
            print(f"FinBERT error: {e}")
            return self.get_sentiment_vader(text)  

    def get_sentiment(self, text):
        text = self.clean_text(text)
        if self.use_transformers:
            return self.get_sentiment_finbert(text)
        else:
            return self.get_sentiment_vader(text)

    def fetch_news_yahoo(self, ticker, days_back=3):
        news_data = []
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', {'class': 'Ov(h) Pend(44px) Pstart(25px)'})

            for item in news_items:
                try:
                    headline = item.find('h3').text
                    timestamp = item.find('span', {'class': 'C(#959595)'}).text
                    date = dt.datetime.now().date()
                    if 'hours' in timestamp or 'minutes' in timestamp:
                        pass  
                    elif 'yesterday' in timestamp.lower():
                        date = date - dt.timedelta(days=1)
                    elif 'days' in timestamp:
                        days = int(re.search(r'(\d+)', timestamp).group(1))
                        date = date - dt.timedelta(days=days)

                    if (dt.datetime.now().date() - date).days <= days_back:
                        news_data.append({'headline': headline, 'date': date})
                except Exception as e:
                    print(f"Error parsing news item: {e}")

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")

        return news_data

    def get_stock_sentiment(self, ticker, days_back=3):
        news = self.fetch_news_yahoo(ticker, days_back)

        if not news:
            return 0 
        sentiments = []
        for item in news:
            sentiment = self.get_sentiment(item['headline'])
            sentiments.append(sentiment)
        return sum(sentiments) / len(sentiments) if sentiments else 0

class SentimentEnhancedTradingEnv:
    @staticmethod
    def enhance_environment(env_class):
        class SentimentTradingEnv(env_class):
            def __init__(self, *args, **kwargs):
                super(SentimentTradingEnv, self).__init__(*args, **kwargs)
                self.sentiment_analyzer = NewsSentimentAnalyzer()
                self.sentiment_history = {symbol: {} for symbol in self.symbols}
                self.sentiment_update_frequency = 7  # days

            def _update_sentiment(self):
                current_date = self.current_date
                if hasattr(current_date, 'date'):
                    current_date = current_date.date()

                for symbol in self.symbols:
                    last_update = max(self.sentiment_history[symbol].keys()) if self.sentiment_history[symbol] else None

                    if last_update is None or (current_date - last_update).days >= self.sentiment_update_frequency:
                        sentiment = self.sentiment_analyzer.get_stock_sentiment(symbol)
                        self.sentiment_history[symbol][current_date] = sentiment

            def _get_observation(self):
                base_observation = super(SentimentTradingEnv, self)._get_observation()
                self._update_sentiment()
                sentiments = []
                for symbol in self.symbols:
                    dates = sorted(self.sentiment_history[symbol].keys())
                    if dates:
                        latest_date = max(dates)
                        sentiments.append(self.sentiment_history[symbol][latest_date])
                    else:
                        sentiments.append(0)  
                sentiment_data = np.tile(sentiments, (base_observation.shape[0], 1))
                enhanced_observation = np.hstack((base_observation, sentiment_data))
                return enhanced_observation

            def step(self, action):
                observation, reward, done, info = super(SentimentTradingEnv, self).step(action)
                info['sentiment'] = {
                    symbol: self.sentiment_history[symbol].get(
                        max(self.sentiment_history[symbol].keys()) if self.sentiment_history[symbol] else None,
                        0
                    ) for symbol in self.symbols
                }
                return observation, reward, done, info
        return SentimentTradingEnv


def create_sentiment_trading_env(stock_dfs):
    enhanced_env_class = SentimentEnhancedTradingEnv.enhance_environment(MultiStockTradingEnv)
    return enhanced_env_class(stock_dfs=stock_dfs)

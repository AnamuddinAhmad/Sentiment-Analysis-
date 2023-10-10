import tweepy
import pandas as pd
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Twitter API credentials (you need to create a Twitter Developer account and obtain these)
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Function to clean tweet text
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment_textblob(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to perform sentiment analysis using VADER
def analyze_sentiment_vader(tweet):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(clean_tweet(tweet))
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Search for tweets related to airlines
search_query = "airline"
tweets = tweepy.Cursor(api.search, q=search_query, lang="en", tweet_mode="extended").items(100)

# Create a DataFrame to store tweet data
data = {'Tweet': [], 'TextBlob_Sentiment': [], 'VADER_Sentiment': []}

for tweet in tweets:
    data['Tweet'].append(tweet.full_text)
    data['TextBlob_Sentiment'].append(analyze_sentiment_textblob(tweet.full_text))
    data['VADER_Sentiment'].append(analyze_sentiment_vader(tweet.full_text))

df = pd.DataFrame(data)

# Analyze and print sentiment statistics
print("Sentiment Analysis using TextBlob:")
print(df['TextBlob_Sentiment'].value_counts())
print("\nSentiment Analysis using VADER:")
print(df['VADER_Sentiment'].value_counts())

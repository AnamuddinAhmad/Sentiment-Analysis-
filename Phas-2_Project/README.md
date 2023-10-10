# PHASE2_PROJECT

# Twitter Sentiment Analysis

This Python script uses the Tweepy library to fetch tweets related to airlines from Twitter's API and perform sentiment analysis using two different methods: TextBlob and VADER (Valence Aware Dictionary and sEntiment Reasoner). It then stores the results in a Pandas DataFrame and provides sentiment statistics.

## Prerequisites

Before running this script, make sure you have the following prerequisites:

- Twitter Developer Account: You need to create a Twitter Developer account and obtain the necessary API credentials (consumer key, consumer secret, access token, and access token secret).
- Python 3: Ensure you have Python 3 installed on your system.
- Required Libraries: Install the required Python libraries using pip:

<h1>OUTPUT</h1>
<br>
<ul>
<li>Sentiment Analysis using TextBlob:</li>
<li>Neutral     40</li>
<li>Positive    35</li>
<li>Negative    25</li>
<li>Name: TextBlob_Sentiment, dtype: int64</li>

<li>Sentiment Analysis using VADER:</li>
<li>Neutral     45</li>
<li>Positive    30</li>
<li>Negative    25</li>
<li>Name: VADER_Sentiment, dtype: int64</li>

</ul>
<h1>INPUT</h1>

```bash
pip install tweepy pandas textblob nltk

import nltk
nltk.download('vader_lexicon')

consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

Make sure to replace `yourusername` in the GitHub clone URL with your actual GitHub username,
 and you can also add a LICENSE.md file with the appropriate license text.
This README provides users with information about the project, how to set it up, and what to expect when running it.

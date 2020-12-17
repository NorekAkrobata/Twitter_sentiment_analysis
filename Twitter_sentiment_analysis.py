import os
import pandas as pd
import matplotlib.pyplot as plt
import twint
import string
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
import nest_asyncio
nest_asyncio.apply()
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', -1)
os.getcwd()

c = twint.Config()
c.To = '@rblz_anders'
c.Limit = 50000
c.Count = True
c.Store_csv = True
c.Output = 'Anders.csv'

#twint.run.Search(c)

df = pd.read_csv('Anders.csv')
df.head()

df = df[['id', 'conversation_id', 'date', 'time', 'user_id', 'tweet','language', 'likes_count', 'hashtags', 'link']]
df.head()

df['language'].value_counts()
df = df[df['language'] == 'en']

tweets = df[['tweet']]

def clean_tweets_tb(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text) #Remove mentions
    text = re.sub(r"_[A-Za-z0-9]+", "", text) #Remove replies
    text = re.sub(r"__", "", text) #Remove replies
    text = re.sub(r'https?://\S+', '', text) #Remove urls
    text = re.sub(' +', ' ', text) #Remove double spaces
    text = "".join([char for char in text if char not in string.punctuation]) #Remove punctuations
    text = text.lower() #Lower text
    
    return text

def clean_tweets_vader(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text) #Remove mentions
    text = re.sub(r"_[A-Za-z0-9]+", "", text) #Remove replies
    text = re.sub(r'https?://\S+', '', text) #Remove urls
    text = re.sub(r"__", "", text) #Remove replies
    text = re.sub(' +', ' ', text) #Remove double spaces
    
    return text

#Vader
vader_analyser = SentimentIntensityAnalyzer() 
    
tweets['tweet_vader'] = tweets['tweet'].apply(clean_tweets_vader)

def vader(text):
    vader = vader_analyser.polarity_scores(text)['compound']
    return vader

tweets['vader'] = tweets['tweet_vader'].apply(vader) 

def sentiment_vader(score):
    if score < -0.05:
        return 'Negative'
    elif (score >= -0.05 and score <= 0.05):
        return 'Neutral'
    else:
        return 'Positive'

tweets['sentiment_vader'] = tweets['vader'].apply(sentiment_vader)

#Textblob

tweets['tweet_tb'] = tweets['tweet'].apply(clean_tweets_tb)

def tb(text):
    pol = TextBlob(text).sentiment.polarity
    return pol

tweets['tb'] = tweets['tweet_tb'].apply(tb)

def sentiment_tb(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

tweets['sentiment_tb'] = tweets['tb'].apply(sentiment_tb)

#Comparison

perc_vader = tweets['sentiment_vader'].value_counts(normalize=True)
perc_tb = tweets['sentiment_tb'].value_counts(normalize=True)
perc_vader, perc_tb = perc_vader*100, perc_tb*100
perc_both = pd.concat([perc_vader, perc_tb], axis=1).reset_index()

fig, ax = plt.subplots()
perc_both.plot.bar(x='index', ax=ax)
ax.set_title('Summary of twitter replies to Anders - divided into 3 groups')
ax.set_xlabel('Type of replies')
ax.set_ylabel('Percentage of total replies')

positive_vader = (tweets[['tweet_vader', 'vader']][tweets['sentiment_vader'] == 'Positive']).sort_values(by=['vader'], ascending=False)
positive_tb = (tweets[['tweet_tb', 'tb']][tweets['sentiment_tb'] == 'Positive']).sort_values(by='tb', ascending=False)
negative_vader = (tweets[['tweet_vader', 'vader']][tweets['sentiment_vader'] == 'Negative']).sort_values(by=['vader'], ascending=True)
negative_tb = (tweets[['tweet_tb', 'tb']][tweets['sentiment_tb'] == 'Negative']).sort_values(by='tb', ascending=True)


#tweets['tweet'] = tweets['tweet'].apply(word_tokenize)








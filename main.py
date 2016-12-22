import tweepy
import string
import csv
import nltk
from tweepy import OAuthHandler
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import lda
import numpy as np
import lda.datasets
import gensim
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
import re
import preprocess


#read_tweets(csv_writer_en)
file = open('help.txt', 'r')
tweets_file = open('tweets.csv', 'a')
lines = file.readlines()
consumer_key = lines[0].rstrip()
consumer_secret = lines[1].rstrip()
access_token = lines[2].rstrip()
access_secret = lines[3].rstrip()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
csv_writer = csv.writer(tweets_file)
csv_reader = csv.reader(open('tweets.csv'))
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
lemmatizer = WordNetLemmatizer()
stop_en = set(stopwords.words('english'))
tweets = []

def read_tweets():
    for row in csv_reader:
        tweets.append(row[0])


#preprocess.fetch_tweets(api, csv_writer)

def clean_tweets(tweets):
    cleaned_tweets = []
    for i in range(0, len(tweets)):
        no_links = re.sub(r"http\S+", "", tweets[i])
        no_RT = no_links.lower().replace('rt', '')
        no_stop = " ".join([i for i in no_RT.lower().split() if i not in stop_en])

        no_punc = ''.join(ch for ch in no_stop if ch not in set(string.punctuation))
        lemmatized = " ".join(lemmatizer.lemmatize(word) for word in no_punc.split())
        print(lemmatized)
        cleaned_tweets.append(lemmatized)

    return cleaned_tweets


read_tweets()
cleaned_tweets = clean_tweets(tweets)

tweet_tokens = [tweet.split() for tweet in cleaned_tweets]


dictionary = corpora.Dictionary(tweet_tokens)
doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in tweet_tokens]

print (doc_term_matrix)
lda = gensim.models.ldamodel.LdaModel
ldamodel = lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))




from nltk import wordpunct_tokenize
import tweepy
import json
import csv
from nltk.corpus import stopwords


def fetch_friends_tweets(api, csv_writer):
    for friend in tweepy.Cursor(api.friends).items(626):
        if (friend.lang == "en"):
            stuff = api.user_timeline(id=friend.id, count=400, include_rts = True, exclude_replies=False)
            for status in stuff:
                if(is_english(status.text)):
                    csv_writer.writerow([status.text])


def fetch_timeline_tweets(api, csv_writer):
    for tweet in tweepy.Cursor(api.home_timeline).items(400):
        if is_english(tweet.text):
            csv_writer.writerow([tweet.text])


def is_english(text):
        languages_ratios = {}
        tokens = wordpunct_tokenize(text)
        words = [word.lower() for word in tokens]

        for language in stopwords.fileids():
            stopwords_set = set(stopwords.words(language))
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)
            languages_ratios[language] = len(common_elements)  # language "score"

        most_rated_language = max(languages_ratios, key=languages_ratios.get)
        if most_rated_language == 'english':
            return True

        return False



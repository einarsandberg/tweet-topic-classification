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
from nltk import word_tokenize,sent_tokenize


#read_tweets(csv_writer_en)
file = open('help.txt', 'r')
tweets_file = open('friends_tweets2.csv', 'a')
lines = file.readlines()
consumer_key = lines[0].rstrip()
consumer_secret = lines[1].rstrip()
access_token = lines[2].rstrip()
access_secret = lines[3].rstrip()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
csv_writer = csv.writer(tweets_file)
csv_reader = csv.reader(open('friends_tweets2.csv'))
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
lemmatizer = WordNetLemmatizer()
stop_en = set(stopwords.words('english'))

def read_tweets():
    count = 0
    tweets = []
    for row in csv_reader:
        # clear out some swedish tweets
        if ("å" not in row[0] and "ä" not in row[0] and "ö" not in row[0]):
            tweets.append(row[0])

    return tweets


def read_my_tweets():
    reader = csv.reader(open("my_tweets.csv"))
    tweets = []
    for row in reader:
        # clear out some swedish tweets
        if ("å" not in row[0] and "ä" not in row[0] and "ö" not in row[0]):
            tweets.append(row[0])

    return tweets


def clean_tweets(tweets):
    cleaned_tweets = []
    for i in range(0, len(tweets)):
        no_links = re.sub(r"http\S+", "", tweets[i])
        no_reply = re.sub('(?<=@)[^\s]+', '', no_links)
        no_reply = no_reply.replace('@', '')
        no_RT = no_reply.lower().replace('rt', '')
        no_amp = no_RT.replace('&amp', '')
        no_stop = " ".join([i for i in no_amp.lower().split() if i not in stop_en])
        no_punc = ''.join(ch for ch in no_stop if ch not in set(string.punctuation))
        lemmatized = " ".join(lemmatizer.lemmatize(word) for word in no_punc.split())
        cleaned_tweets.append(lemmatized)

    return cleaned_tweets


def get_document_topics(tweets, ldamodel, doc_term_matrix):
    categorized_tweets = []
    categorized_tweet = {}

    for i in range(0, len(tweets)):
        topics = ldamodel.get_document_topics(doc_term_matrix[i])
        sorted_by_value = sorted(topics, key=lambda tup: tup[1], reverse=True)
        categorized_tweet = {"processed_tweet": tweets[i], "topic_id": sorted_by_value[0][0],
                             "topic_words": ldamodel.show_topic(sorted_by_value[0][0], topn=3),
                             "topic_probability": sorted_by_value[0][1]}
        categorized_tweets.append(categorized_tweet)


    print_similar_tweets(categorized_tweets)
    return categorized_tweets


def print_similar_tweets(tweets):
    # Go through all tweets and print them to files named "topic" + topicID
    topic_ids = []
    for i in range(0, len(tweets)):
        with open("topic"+str(tweets[i]["topic_id"])+".csv", 'a') as csv_file:
            writer = csv.writer(csv_file)
            print(tweets[i]["processed_tweet"])
            if (tweets[i]["topic_id"] not in topic_ids):
                writer.writerow(tweets[i]["topic_words"])
                topic_ids.append(tweets[i]["topic_id"])

            writer.writerow([tweets[i]["processed_tweet"]])


def extract_nouns(tweets):
    nouns_list = []
    for tweet in tweets:
        sentences = nltk.sent_tokenize(tweet)
        nouns = []
        for sentence in sentences:
            for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
                if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                    nouns.append(word)

        nouns_list.append(nouns)

    return nouns_list


def save_categorized_tweets(tweets, file_name):
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list(tweets[0].keys()))
        for tweet in tweets:
            writer.writerow(list(tweet.values()))


def train_model():
    tweets = read_tweets()
    cleaned_tweets = clean_tweets(tweets)
    nouns = extract_nouns(cleaned_tweets)
    dictionary = corpora.Dictionary(nouns)
    dictionary.filter_extremes(no_below=20, no_above=0.7)
    doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in nouns]
    lda = gensim.models.ldamodel.LdaModel
    #test alpha = 0.5
    ldamodel = lda(doc_term_matrix, num_topics=40, alpha=0.001, id2word=dictionary, passes=50)
    ldamodel.save('model.lda')
    dictionary.save("dictionary.dict")
    categorized_tweets = get_document_topics(cleaned_tweets, ldamodel, doc_term_matrix)
    save_categorized_tweets(categorized_tweets, "trained_categorized_tweets.csv")


def run_model():
    tweets = read_my_tweets()
    cleaned_tweets = clean_tweets(tweets)
    nouns_list = extract_nouns(cleaned_tweets)
    ldamodel = gensim.models.LdaModel.load('model.lda')
    dictionary = corpora.Dictionary.load("dictionary.dict")
    new_ldas = []
    for nouns in nouns_list:
        new_bow = dictionary.doc2bow(nouns)
        new_ldas.append(ldamodel[new_bow])


    categorized_tweets = get_document_topics(cleaned_tweets, ldamodel, new_ldas)
    save_categorized_tweets(categorized_tweets, "my_categorized_tweets.csv")





#preprocess.fetch_friends_tweets(api, csv_writer)


#train_model()
run_model()

# Testa att skriva ut alla tweets för samma topic.
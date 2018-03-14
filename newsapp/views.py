from django.shortcuts import render

import json
import urllib.request
from newsapi import NewsApiClient
from nltk.corpus import stopwords
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from functools import partial

API_KEY = "87e663cbe74e4c0c9a8cb4725bce4b42"

# Create your views here.



def train_model():
    perform_preprocessing()

garbage = set(stopwords.words('english'))
vectorizer = pickle.load(open("/home/mandeep/git_test/newsmaster/newsapp/vectorizer.p", "rb"))
encoder = pickle.load(open("/home/mandeep/git_test/newsmaster/newsapp/encoder.p", "rb"))
keywords = pickle.load(open("/home/mandeep/git_test/newsmaster/newsapp/keywords.p", "rb"))
classifier = pickle.load(open("/home/mandeep/git_test/newsmaster/newsapp/classifier.p", "rb"))

def normalize_text(s, keywords):
    s = s.lower()

    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    s = re.sub('\s+', ' ', s)
    s = s.split()
    s = list(filter((lambda x: x not in garbage), s))
    for word in s:
        keywords.append(word)
    s = " ".join(s)
    return s

def format_live_news_title(title, keywords):
    return [" ".join(list(word for word in title.split() if word in keywords))]


def find_category(title, vectorizer, encoder, keywords, classifier):
    sample = normalize_text(title, [])
    sample = format_live_news_title(sample, keywords)
    sample = vectorizer.transform(sample)
    output = classifier.predict(sample)
    return encoder.inverse_transform(output)[0]

def get_news_articles(category):
    api = NewsApiClient(api_key=API_KEY)
    articles_dict = api.get_everything(sources='google-news')

    articles = articles_dict['articles']
    # print(articles)
    for article in articles:
        article["category"] = find_category(article["title"], vectorizer, encoder, keywords, classifier)

    relevant_articles = list(filter((lambda x : category == x["category"]), articles))
    return relevant_articles


def newsfeed(request):
    # print("REMOVE THIS ONCE DEBUGGING IS DONE!!!\n")
    category = 't'
    if(request.GET.get('techbtn')):
        category = 't'
    elif(request.GET.get('medicalbtn')):
        category = 'm'
    elif(request.GET.get('businessbtn')):
        category = 'b'
    elif(request.GET.get('enterbtn')):
        category = 'e'

    articles = get_news_articles(category)
    # print('articles :: ',articles)

    # args['articles'] = articles
    article_dict = {}
    for article in articles:
        article_dict[str(article['title'])] = (str(article['url']), str(article['publishedAt']))

    return render(request, 'newsapp/newsfeed.html', {'article_list': article_dict})

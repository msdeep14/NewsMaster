# import for training model
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from tkinter import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

from sklearn.linear_model import SGDClassifier
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
# import bayes

# stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# global
ps = PorterStemmer()
garbage = set(stopwords.words('english'))

# perform filtering, tokenization and finally
# stop words removal
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

# get root word using stemming
def perform_stemming(keywords):
    result_set = []
    for w in keywords:
        result_set.append(ps.stem(w))
    return result_set

def train_model(news, keywords):
    news['category'] = news['category'].fillna('x')
    vectorizer = CountVectorizer()

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=10, max_features=10000, ngram_range=(1, 2))
    vz = vectorizer.fit_transform(list(news['title']))
    print(vz.shape) # give no of dimensions(value of 2nd index)

    x = vectorizer.fit_transform(news['title'])

    x = TfidfTransformer().fit_transform(x)
    encoder = LabelEncoder()
    y = encoder.fit_transform(news['category'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # classification using naive bayes
    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    # classification using svm classifier
    text_clf_svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
    text_clf_svm = text_clf_svm.fit(x_train, y_train)

    print("\nTraining complete")
    print("\nSVM Accuracy:: ",text_clf_svm.score(x_test,y_test))

    print("\nNaive Bayes Accuracy:: ", nb.score(x_test, y_test))

    pickle.dump(vectorizer, open("pickle-data/vectorizer.p", "wb"))
    pickle.dump(encoder, open("pickle-data/encoder.p", "wb"))
    pickle.dump(keywords, open("pickle-data/keywords.p", "wb"))
    pickle.dump(nb, open("pickle-data/classifier.p", "wb"))
    print("Model saved")


stop = set(stopwords.words('english'))
from string import punctuation

def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]

        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``',
                                            u'\u2014', u'\u2026', u'\u2013'], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    except Exception as e:
        print(e)

def get_keywords(category,news):
    tokens = news[news['category'] == category]['tokens']
    alltokens = []
    counter = 0
    try:
        for token_list in tokens:
            alltokens += token_list
        counter = Counter(alltokens)
        return counter.most_common(10)
    except:
        counter = Counter(alltokens)
        return counter.most_common(10)


def perform_preprocessing():
    # news = pd.read_excel('newss.xlsx', usecols=['title', 'category'])
    # news = pd.read_excel('newss.xlsx', usecols=["B,C"])
    news = pd.read_excel('newsapp/newss.xlsx')

    # plot the no of headlines per each category
    label = news['category'].value_counts().keys().tolist()
    counts = news['category'].value_counts().tolist()
    index = np.arange(len(label))
    plt.bar(index, counts)
    plt.xlabel('news category', fontsize=5)
    plt.ylabel('no of news headlines', fontsize=5)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title('No of news headlines for each category')
    plt.show()

    # remove duplicate title columns
    news = news.drop_duplicates('title')

    # remove rows with empty titles
    news = news[~news['title'].isnull()]

    news['tokens'] = news['title'].map(tokenizer)
    # print('news tokens:: ',news['tokens'])
    for descripition, tokens in zip(news['title'].head(5), news['tokens'].head(5)):
        print('description:', descripition)
        print('tokens:', tokens)
        print()

    for category in set(news['category']):
        print('category :', category)
        print('top 10 keywords:', get_keywords(category,news))
        print('---')


    keywords = []
    news['title'] = [normalize_text(str(s), keywords) for s in news['title']]
    keywords = set(keywords)

    s1 = len(keywords)

    keywords = perform_stemming(keywords)
    keywords = set(keywords)

    s2 = len(keywords)
    print("total no of keywords before and after stemming\ns1 :: ",s1,"s2:: ",s2)

    train_model(news, keywords)


# function to train model at system start
#perform_preprocessing()

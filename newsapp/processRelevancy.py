'''
title: 'Russian presidential elections, Indian Wells and other news in pictures':
url: ('http://www.thehindu.com/news/russian-presidential-elections-indian-wells-and-other-news-in-pictures/article23290204.ece',
publishedAt: 'Mar 19, 2018',
source: 'The Hindu',
description: 'Published at 8:30 a.m\xa0Juan Martin del Potro saved three match points in a thrilling final at the BNP Paribas Open before handing world number one Roger Federer his first loss of the year and claiming')
'''

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date
import datetime
import operator
from collections import OrderedDict

technology = {'techcrunch':10, 'the-next-web':9, 'wired':8, 'mashable':7, 'the-verge':6,'techradar':5, 'business-insider':4}
business = {'business-insider':10, 'bloomberg':9, 'the-wall-street-journal':8, 'cnbc':7, 'financial-times':6,
            'financial-post':5}
entertainment = {'entertainment-weekly':10, 'mtv-news':9, 'mtv-news-uk':8,'vice-news':7}
medical = {'medical-news-today':10}

def get_source_score(article_value, category):
    src = article_value[4]
    # print("src:: ", src, "  ", category)
    var = 0
    if(category == 't'):
        try:
            var = technology[src]
        except:
            pass
    elif(category == 'm'):
        try:
            var = medical[src]
        except:
            pass
    elif(category == 'b'):
        try:
            var = business[src]
        except:
            pass
    elif(category == 'e'):
        try:
            var = entertainment[src]
        except:
            pass
    return var

def get_sentiment_score(sentence):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)

    # for k in sorted(ss):
    #     print('{0}: {1}, '.format(k, ss[k]), end='')
    try:
        return ss['compound']
    except:
        return 0


def get_score(article, article_value, category):
    # print("article:: ",article, "\n",article_value)
    #print(article_value[1],"  ",datetime.date.today(), "  ", article_value[5])

    dt = article_value[5].split('-')
    dt1 = date(int(dt[0]), int(dt[1]), int(dt[2]))
    dt2 = datetime.date.today()
    age = (dt2 - dt1).days
    source = get_source_score(article_value, category)
    # print("source score :: ", source)
    sentiment = get_sentiment_score(article_value[3])
    return (age + source + sentiment)

def get_relevant_articles(article_dict, category):
    # print("articles123:: \n",articles)
    # sort the articles according to source
    relevant_articles = {}
    for article, article_value in article_dict.items():
        relevant_articles[article] = get_score(article, article_value, category)

    # print("relevant articles :: \n",relevant_articles)
    sorted_articles = sorted(relevant_articles.items(), key=operator.itemgetter(1))

    # sort article_dict according to sorted_articles
    # print("sorted_articles :: ", reversed(sorted_articles), "\n")
    sorted_article_dict = OrderedDict()
    for ar in reversed(sorted_articles):
        sorted_article_dict[ar[0]] = article_dict[ar[0]]

    # TODO: sort according to age+source and again sort according to
    # polarity of article

    return sorted_article_dict

'''
title: 'Russian presidential elections, Indian Wells and other news in pictures':
url: ('http://www.thehindu.com/news/russian-presidential-elections-indian-wells-and-other-news-in-pictures/article23290204.ece',
publishedAt: 'Mar 19, 2018',
source: 'The Hindu',
description: 'Published at 8:30 a.m\xa0Juan Martin del Potro saved three match points in a thrilling final at the BNP Paribas Open before handing world number one Roger Federer his first loss of the year and claiming')
'''

technology = {'techcrunch':1, 'the-next-web':2, 'wired':3, 'mashable':4, 'the-verge':5,'techradar':6}
business = {'business-insider':1, 'bloomberg':2, 'the-wall-street-journal':3, 'cnbc':4, 'financial-times':5,
            'financial-post':6}
entertainment = {'entertainment-weekly':1, 'mtv-news':2, 'mtv-news-uk':3,'vice-news':4}
medical = {'medical-news-today':1}




def get_relevant_articles(articles, category):
    # print("articles:: \n",articles)
    # sort the articles according to source
    
    try:
        articles = sorted(articles, key = lambda k: k.get('source', 0))
    except AttributeError:
        pass
    return articles

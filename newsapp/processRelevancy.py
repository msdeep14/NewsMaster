'''
0  ('http://techcrunch.com/2018/03/22/alphabets-outline-lets-you-build-your-own-vpn/',
1 'Mar 22, 2018',
2  'TechCrunch',
3 'Alphabet’s cybersecurity division Jigsaw released an interesting new project called Outline. If I simplify things quite a lot, it lets anyone create and run a VPN server on DigitalOcean, and then grant your team access to this server. I played a bit with Outl…',
4  'techcrunch',
5	 '2018-03-22',
6	 'Alphabet’s Outline lets you build your own VPN')
'''
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date
import datetime
import operator
from collections import OrderedDict

from . import preprocessing

technology = {'techcrunch':10, 'the-next-web':9, 'wired':8, 'mashable':7, 'the-verge':6,'techradar':5, 'business-insider':4
                ,'ars-technica':4,'engadget':3,'recode':2}
business = {'business-insider':10, 'bloomberg':9, 'the-wall-street-journal':8, 'cnbc':7, 'financial-times':6,
            'financial-post':5,'australian-financial-review':4,'fortune':3,'the-economist':2}
entertainment = {'entertainment-weekly':10, 'mtv-news':9, 'mtv-news-uk':8,'vice-news':7,'buzzfeed':6,
                'daily-mail':5,'the-lad-bible':4,'polygon':3}
medical = {'medical-news-today':10}

def get_source_score(article_value, category):
    src = article_value[4]
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
    try:
        return ss['compound']
    except:
        return 0

'''
NE Type	             Examples
ORGANIZATION	Georgia-Pacific Corp., WHO
PERSON	        Eddy Bonte, President Obama
LOCATION	    Murray River, Mount Everest
DATE	        June, 2008-06-29
TIME	        two fifty a m, 1:30 p.m.
MONEY	        175 million Canadian Dollars, GBP 10.40
PERCENT	        twenty pct, 18.75 %
FACILITY	    Washington Monument, Stonehenge
GPE	            South East Asia, Midlothian
'''

def get_named_entities(sentence):
    named_entities = []
    for sent in nltk.sent_tokenize(sentence):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                # print(chunk.label(), ' '.join(c[0] for c in chunk))
                s = {chunk.label(), ' '.join(c[0] for c in chunk)}
                named_entities.append(s)
    return named_entities

def get_text_quality_score(sentence):
    if sentence == '':
        return 0
    else:
        prev_score = len(sentence.split(' '))
        keywords = []
        after = preprocessing.normalize_text(sentence, keywords)
        after_score = len(after.split(' '))
        # print("prev, after score:: ", prev_score, "  ", after_score)
        return (after_score/prev_score)

def get_score(article, article_value, category):
    # print("article:: ",article, "\n",article_value)
    #print(article_value[1],"  ",datetime.date.today(), "  ", article_value[5])

    dt = article_value[5].split('-')
    dt1 = date(int(dt[0]), int(dt[1]), int(dt[2]))
    dt2 = datetime.date.today()
    age_score = (dt2 - dt1).days
    source_score = get_source_score(article_value, category)
    # print("source score :: ", source)
    sentiment_score = get_sentiment_score(article_value[3])
    text_quality_score = get_text_quality_score(article_value[3])
    named_entities = get_named_entities(article_value[3])
    return (age_score + source_score + sentiment_score + text_quality_score)

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

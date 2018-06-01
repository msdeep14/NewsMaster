import calendar
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.shortcuts import render
from newsapp.forms import UserForm
from django.contrib.auth import logout
from newsapp.models import Users, User
from django.core.mail import send_mail

from django.shortcuts import render

import json
import urllib.request
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

from dateutil.parser import parse
from datetime import datetime, timedelta

from . import preprocessing
from . import processRelevancy

API_KEY = "87e663cbe74e4c0c9a8cb4725bce4b42"
# from newsapi.newsapi_client import NewsApiClient
# from newsapp.newsapi.newsapi_client import NewsApiClient
from . import newsapi
# Create your views here.

garbage = set(stopwords.words('english'))
vectorizer = pickle.load(open("pickle-data/vectorizer.p", "rb"))
encoder = pickle.load(open("pickle-data/encoder.p", "rb"))
keywords = pickle.load(open("pickle-data/keywords.p", "rb"))
classifier = pickle.load(open("pickle-data/classifier.p", "rb"))


def format_live_news_title(title, keywords):
    return [" ".join(list(word for word in title.split() if word in keywords))]


def find_category(title, vectorizer, encoder, keywords, classifier):
    sample = preprocessing.normalize_text(title, [])
    sample = format_live_news_title(sample, keywords)
    sample = vectorizer.transform(sample)
    output = classifier.predict(sample)
    return encoder.inverse_transform(output)[0]

def get_news_articles(category):
    api = newsapi.newsapi_client.NewsApiClient(api_key=API_KEY)
    from_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
    if category == 'top':
        articles_dict = api.get_top_headlines(language='en',page_size=100,page=5)
        return articles_dict
    else:
        articles_dict = api.get_everything(sources='techcrunch,the-next-web,wired,the-lad-bible,polygon,fortune,the-economist,ars-technica,engadget,recode,australian-financial-review,mashable,the-verge,techradar,bloomberg,the-wall-street-journal,cnbc,financial-times,financial-post,mtv-news,mtv-news-uk,vice-news,medical-news-today,business-insider,cnn,entertainment-weekly,the-hindu,msnbc',language='en',page_size=100,from_parameter=from_date,page=5)
    # print("articles:: ",articles_dict)
    # print("health sources:: ",api.get_sources(language='en', category='health'))
    # print("entertainment sources:: ",api.get_sources(language='en', category='entertainment'))
    # print("business sources:: ",api.get_sources(language='en', category='business'))
    # print("technology sources:: ",api.get_sources(language='en', category='technology'))
    filtered_articles = []
    try:
        articles = articles_dict['articles']
        for article in articles:
            article["category"] = find_category(article["title"], vectorizer, encoder, keywords, classifier)

        filtered_articles = list(filter((lambda x : category == x["category"]), articles))

    # only 1000 requests are allowed per day to newsapi.org
    except:
        print("{'message': 'You have exhausted your daily request limit. Developer accounts are limited to 1,000 requests in a 24 hour period. Please upgrade to a paid plan if you need more requests.', 'status': 'error', 'code': 'rateLimited'}")
    # print("len :: ",len(articles))
    # print(articles)

    return filtered_articles

@login_required
def newsfeed(request):
    # print("REMOVE THIS ONCE DEBUGGING IS DONE!!!\n")
    category = 'top'
    category_value = 'Top Stories'
    if(request.GET.get('topbtn')):
        category = 'top'
        category_value = 'Top Stories'
    elif(request.GET.get('techbtn')):
        category = 't'
        category_value = 'Technology'
    elif(request.GET.get('medicalbtn')):
        category = 'm'
        category_value = 'Medical'
    elif(request.GET.get('businessbtn')):
        category = 'b'
        category_value = 'Business'
    elif(request.GET.get('enterbtn')):
        category = 'e'
        category_value = 'Entertainment'

    articles = get_news_articles(category)
    # print('articles :: ',articles)

    # TODO: get user information and show top stories according
    # to user's interest
    print("USER :: ",request.user,"\n")


    article_dict = {}
    if(category == 'top'):
        # print(articles['articles'])
        for article in articles['articles']:
            dt = parse(str(article['publishedAt']))
            dt1 = dt.strftime('%h %d, %Y')
            dt2 = dt.strftime('%Y-%m-%d')
            article_dict[str(article['title'])] = (str(article['url']), str(dt1),
            str(article['source']['name']), str(article['description']), str(article['source']['id']),str(dt2), str(article['title']),str(category_value))
        return render(request, 'newsapp/newsfeed.html', {'article_list': article_dict})
    else:
        for article in articles:
            dt = parse(str(article['publishedAt']))
            dt1 = dt.strftime('%h %d, %Y')
            dt2 = dt.strftime('%Y-%m-%d')
            article_dict[str(article['title'])] = (str(article['url']), str(dt1),
            str(article['source']['name']), str(article['description']), str(article['source']['id']),str(dt2), str(article['title']),str(category_value))

        # TODO: now process articles for relevancy
        sorted_article_dict = processRelevancy.get_relevant_articles(article_dict, category)
        # print(sorted_article_dict)
        return render(request, 'newsapp/newsfeed.html', {'article_list': sorted_article_dict})

def register(request):
    context = RequestContext(request)
    registered = False
    if request.method == 'POST':
        user_form = UserForm(data = request.POST)
        if user_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()
            registered = True
        else:
            print(user_form.errors)
    else:
        user_form = UserForm()
    return render(request,'newsapp/register.html',{'user_form':user_form, 'registered':registered})



def user_login(request):
	context = RequestContext(request)
	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(username=username, password=password)
		if user:
			if user.is_active:
				login(request, user)
				return HttpResponseRedirect('/')
			else:
				return HttpResponse("Your NewsMaster account is disabled.")
		else:
			print("Invalid login details: {0}, {1}".format(username, password))
			return HttpResponse("Invalid login details supplied.")
	else:
		return render(request,'newsapp/login.html', {})


@login_required
def user_logout(request):
    logout(request)
    return HttpResponseRedirect('/')



'''
newsapi sources

abc-news
abc-news-au
aftenposten
al-jazeera-english
ansa
argaam
ars-technica
ary-news
associated-press
australian-financial-review
axios
bbc-news
bbc-sport
bild
blasting-news-br
bleacher-report
bloomberg
breitbart-news
business-insider
business-insider-uk
buzzfeed
cbc-news
cbs-news
cnbc
cnn
cnn-es
crypto-coins-news
daily-mail
der-tagesspiegel
die-zeit
el-mundo
engadget
entertainment-weekly
espn
espn-cric-info
financial-post
financial-times
focus
football-italia
fortune
four-four-two
fox-news
fox-sports
globo
google-news
google-news-ar
google-news-au
google-news-br
google-news-ca
google-news-fr
google-news-in
google-news-is
google-news-it
google-news-ru
google-news-sa
google-news-uk
goteborgs-posten
gruenderszene
hacker-news
handelsblatt
ign
il-sole-24-ore
independent
infobae
info-money
la-gaceta
la-nacion
la-repubblica
le-monde
lenta
lequipe
les-echos
liberation
marca
mashable
medical-news-today
metro
mirror
msnbc
mtv-news
mtv-news-uk
national-geographic
nbc-news
news24
new-scientist
news-com-au
newsweek
new-york-magazine
next-big-future
nfl-news
nhl-news
nrk
politico
polygon
rbc
recode
reddit-r-all
reuters
rt
rte
rtl-nieuws
sabq
spiegel-online
svenska-dagbladet
t3n
talksport
techcrunch
techcrunch-cn
techradar
the-economist
the-globe-and-mail
the-guardian-au
the-guardian-uk
the-hill
the-hindu
the-huffington-post
the-irish-times
the-lad-bible
the-new-york-times
the-next-web
the-sport-bible
the-telegraph
the-times-of-india
the-verge
the-wall-street-journal
the-washington-post
time
usa-today
vice-news
wired
wired-de
wirtschafts-woche
xinhua-net
ynet

'''

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
import bayes

# stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


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
    #print("s :: ",s)
    return s

# get root word using stemming
def perform_stemming(keywords):
    result_set = []
    for w in keywords:
        result_set.append(ps.stem(w))
    return result_set

def perform_preprocessing():
    ps = PorterStemmer()


    # Read data and initialize stop-words
    # news = pd.read_excel('newss.xlsx', usecols=['title', 'category'])
    # news = pd.read_excel('newss.xlsx', usecols=["B,C"])
    garbage = set(stopwords.words('english'))

    news = pd.read_csv('/home/mandeep/news.csv')
    news2 = pd.read_excel('newss.xlsx')

    # print("normalized\n")
    # print(news)
    print("stopwords:: ",garbage)
    # fp1 = open("/home/mandeep/Downloads/project-be/stopwords.txt",'w')
    # fp1.write(str(stopwords))



    keywords = []
    news['title'] = [normalize_text(str(s), keywords) for s in news['title']]
    keywords = set(keywords)

    s1 = len(keywords)

    keywords = perform_stemming(keywords)
    keywords = set(keywords)

    s2 = len(keywords)
    print("s1 :: ",s1,"s2:: ",s2)

    # print("keywords:: ",keywords)
    fp = open("/home/mandeep/project-be/keywords.txt",'w')
    fp.write(str(keywords))


    news['category'] = news['category'].fillna('x')

    # print(news['title'])

    vectorizer = CountVectorizer()

    # print("vectorizer:: ",vectorizer)
    x = vectorizer.fit_transform(news['title'])

    # print("Events Clustering data")
    # coreLabels, coreSamples = dbscanClustering(x)

    # print("coreLabels:: ",coreLabels,"\n\ncoresamples:: ",coreSamples)

    # print("\nx:: ",x)
    # TODO: Check if TF-IDF actually helps
    x = TfidfTransformer().fit_transform(x)
    # print("\nxtfid:: ",x)
    encoder = LabelEncoder()

    # print("\nencoder:: ",encoder)
    y = encoder.fit_transform(news['category'])

    # print("\ny:: ",y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    # bayes.main_naive(news)

    # print("\nnb:: ",nb)

    # from pprint import pprint
    # pprint(vars(nb))

    print("Training complete")
    print("Accuracy ", nb.score(x_test, y_test))

    pickle.dump(vectorizer, open("vectorizer.p", "wb"))
    pickle.dump(encoder, open("encoder.p", "wb"))
    pickle.dump(keywords, open("keywords.p", "wb"))
    pickle.dump(nb, open("classifier.p", "wb"))
    print("Model saved")

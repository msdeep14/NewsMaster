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
# import bayes

# stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

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

    # print("vectorizer:: ",vectorizer)
    x = vectorizer.fit_transform(news['title'])

    # print("Events Clustering data")
    # coreLabels, coreSamples = dbscanClustering(x)

    # print("coreLabels:: ",coreLabels,"\n\ncoresamples:: ",coreSamples)

    # print("\nx:: ",x)
    # TODO: Check if TF-IDF actually helps
    x = TfidfTransformer().fit_transform(x)

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from sklearn.cluster import MiniBatchKMeans

    num_clusters = 30
    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
    init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
    kmeans = kmeans_model.fit(x)
    kmeans_clusters = kmeans.predict(x)
    kmeans_distances = kmeans.transform(x)

    sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d:" % i)
        aux = ''
        for j in sorted_centroids[i, :10]:
            aux += terms[j] + ' | '
        print(aux)
        print()

    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
    colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
    "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
    "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
    "#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

    import bokeh.plotting as bp
    from bokeh.models import HoverTool, BoxSelectTool
    from bokeh.plotting import figure, show, output_notebook
    plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the news",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None, y_axis_type=None, min_border=1)

    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['title'] = news['title']
    kmeans_df['category'] = news['category']

    plot_kmeans.scatter(x='x', y='y',color=colormap[kmeans_clusters],source=kmeans_df)
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover.tooltips={"title": "@title", "category": "@category", "cluster":"@cluster"}
    show(plot_kmeans)



    # for (i, desc),category in zip(enumerate(news.description),news['category']):
    #     if(i < 5):
    #         print("Cluster " + str(kmeans_clusters[i]) + ": " + desc +
    #           "(distance: " + str(kmeans_distances[i][kmeans_clusters[i]]) + ")")
    #     print('category: ',category)
    #     print('---')


    # print("\nxtfid:: ",x)
    encoder = LabelEncoder()

    # print("\nencoder:: ",encoder)
    y = encoder.fit_transform(news['category'])

    # print("\ny:: ",y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # classification using naive bayes
    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    # classification using svm classifier
    text_clf_svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
    text_clf_svm = text_clf_svm.fit(x_train, y_train)

    print("Training complete")
    print("svm Accuracy:: ",text_clf_svm.score(x_test,y_test))

    # bayes.main_naive(news)

    # print("\nnb:: ",nb)

    # from pprint import pprint
    # pprint(vars(nb))

    print("Naive bayes Accuracy ", nb.score(x_test, y_test))

    pickle.dump(vectorizer, open("pickle-data/vectorizer.p", "wb"))
    pickle.dump(encoder, open("pickle-data/encoder.p", "wb"))
    pickle.dump(keywords, open("pickle-data/keywords.p", "wb"))
    pickle.dump(nb, open("pickle-data/classifier.p", "wb"))
    print("Model saved")

def perform_preprocessing():
    # news = pd.read_excel('newss.xlsx', usecols=['title', 'category'])
    # news = pd.read_excel('newss.xlsx', usecols=["B,C"])
    news = pd.read_excel('newsapp/newss.xlsx')
    # news2 = pd.read_excel('newss.xlsx')

    # print(news)
    # fp1 = open("/home/mandeep/Downloads/project-be/stopwords.txt",'w')
    # fp1.write(str(stopwords))

    keywords = []
    news['title'] = [normalize_text(str(s), keywords) for s in news['title']]
    keywords = set(keywords)

    s1 = len(keywords)

    keywords = perform_stemming(keywords)
    keywords = set(keywords)

    s2 = len(keywords)
    # print("s1 :: ",s1,"s2:: ",s2)

    # print("keywords:: ",keywords)
    # fp = open("/home/mandeep/project-be/keywords.txt",'w')
    # fp.write(str(keywords))

    train_model(news, keywords)


# function to train model at system start
#perform_preprocessing()

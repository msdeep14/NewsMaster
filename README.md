# NewsMaster

Get top news based on content relevancy in 5 categories(Top Stories, Entertainment, Technology, Business, Health)

![business news](https://github.com/msdeep14/NewsMaster/blob/master/screenshots/business-news.png)

## Project Development

The project is developed in Python-Django framework, you can read complete implementation of project in documentation folder.

## Basic Idea

  1. Fetch news of various sources from https://newsapi.org/ 
  2. Categorize the news using naive bayes classifier(technology, entertainment, business, health)
  3. Sort the news according to content of news headlines and display on web browser
  
  Content Selection Factors:
  1. Age of article(publication date)
  2. Source Quality
  3. Text Quality
  4. Sentiment Score
  
## Machine Learning algorithms

Implemented training model using Naive bayes, SVM and deep learning model using keras for the dataset, got highest accuracy for naive bayes classifier(0.89)

## Dataset
  
  1. [News headlines of India](https://www.kaggle.com/therohk/india-headlines-news-dataset)
  2. [News Aggregator Dataset](https://www.kaggle.com/uciml/news-aggregator-dataset)

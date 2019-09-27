from flask import Flask, request
from flask import render_template
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
vectorizer = CountVectorizer(stop_words="english", analyzer='word', lowercase= True)
tfidf_transformer = TfidfTransformer()

import numpy as np
import pandas as pd
import time

start = time.time()
dataset = fetch_20newsgroups(shuffle=True, download_if_missing=False)
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)

vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

transform = tfidf_transformer.fit_transform(vectors)
transform_test = tfidf_transformer.transform(vectors_test)

clf = MultinomialNB(alpha=.01)
clf.fit(transform, newsgroups_train.target)
pred = clf.predict(transform_test)
end = time.time()

main_dataset = pd.DataFrame({'categories' : dataset.target, 'contents' : dataset.data})
train_dataset = pd.DataFrame({'categories' : newsgroups_train.target, 'contents' : newsgroups_train.data})
test_dataset = pd.DataFrame({'categories' : newsgroups_test.target, 'contents' : newsgroups_test.data})
prediction = pd.DataFrame({'Comments': newsgroups_test.data, 'Actual': newsgroups_test.target, 'predicted': pred})

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello World Chinedu'

@app.route('/hello')
def new_route():
  return 'testing new routes'


@app.route('/dashboard', methods=['POST', 'GET'])
def login():
  pred = None
  if request.method == 'POST':
    success = request.form['text']

    vectors = vectorizer.fit_transform(newsgroups_train.data)
    vectors_test = vectorizer.transform([success])

    transform = tfidf_transformer.fit_transform(vectors)
    transform_test = tfidf_transformer.transform(vectors_test)

    clf = MultinomialNB(alpha=.01)
    clf.fit(transform, newsgroups_train.target)
    pred = clf.predict(transform_test)
    
  return render_template('dashboard.html', pred = pred)


@app.route('/dataset')
def dataset():

  return render_template('dataset.html', main = main_dataset, test = test_dataset,  train = train_dataset)

@app.route('/categories')
def categories():

  categories_all = fetch_20newsgroups(subset='all')
  main_dataset = pd.DataFrame({'categories' : categories_all.target_names})
  return render_template('categories.html', categories = main_dataset )

@app.route('/statistics')
def statistics():
  dataset = fetch_20newsgroups(shuffle=True, download_if_missing=False)
  newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
  newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)  
  len_dataset = len(dataset.data)
  len_train = newsgroups_train.target.shape
  len_test = newsgroups_test.target.shape
  categories = len(dataset.target_names)
  met = float("%0.2f" % (metrics.f1_score(newsgroups_test.target, pred, average='macro') * 100))
  time = float("%0.2f" % (end - start))

  return render_template('statistics.html', len = len_dataset, lentrain = len_train,  lentest = len_test, categories = categories , accuracy = met, duration = time , cl = classification)

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username

@app.route('/home')
@app.route('/home/<username>')
def home(name=None):
  return render_template('home.html', name=name)
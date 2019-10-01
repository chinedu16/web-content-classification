from flask import Flask, request
from flask import render_template
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn import metrics
import requests
from bs4 import BeautifulSoup
import urllib
import urllib.request
vectorizer = CountVectorizer(stop_words='english', analyzer='word', lowercase=True)
tfidf_transformer = TfidfTransformer()

import numpy as np
import pandas as pd
import time

# CSV dataset
csv_dataset = pd.read_csv('dataset.csv', error_bad_lines=False)

# split dataset
X = csv_dataset['category']
Y = csv_dataset['headline']

validation_size = 0.25
seed = 20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# X_train is the Category of the train data, Y_train is the short description of the train data 
# X_test is the Category of the test data, Y_test is the short description of the test data

# time start
start = time.time()

# feature extractin from train data
train_dataset_count_sd = vectorizer.fit_transform(Y_train.apply(lambda x: np.str_(x)))
X_train_tfidf = tfidf_transformer.fit_transform(train_dataset_count_sd)

# feature extraction for test data
test_dataset_count_sd = vectorizer.transform(Y_test.values.astype('U')) 
X_test_tfidf = tfidf_transformer.transform(test_dataset_count_sd)

# define multinomial naive bayes 
multi_naive = MultinomialNB()

# train data using algorithm
multi_naive.fit(X_train_tfidf, X_train)

# prediction
naive = multi_naive.predict(X_test_tfidf)

# end time
stop = time.time()

# dataframes for display
csv_dataset = pd.DataFrame({'Categories': X, 'Short Description': Y})
csv_train = pd.DataFrame({'Categories': X_train, 'Short Description': Y_train})
csv_test = pd.DataFrame({'Categories': X_test, 'Short Description': Y_test})
csv_prediction = pd.DataFrame({'Short Description': Y_test, 'Actual': X_test, 'predicted': naive})

# reporting and statistics
cmc = metrics.classification_report(X_test, naive)
csv_metrics_dataframe = cmc

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello World Chinedu'

@app.route('/hello')
def new_route():
  return 'testing new routes'


@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
  pred = None
  if request.method == 'POST':
    if "submit_url" in request.form:
      url_success = request.form['url_text']
      url_art = url_success
      html = urllib.request.urlopen(url_art).read()
      soup = soup = BeautifulSoup(html, "html.parser")
      for script in soup(["script", "style"]):
        script.extract()
      text = soup.get_text()
      lines = (line.strip() for line in text.splitlines())
      chunks = (phrase.strip() for line in lines for phrase in line.split(" ")) 
      text = '\n'.join(chunk for chunk in chunks if chunk)

      vectors_test = vectorizer.transform([text,])
      transform_test = tfidf_transformer.transform(vectors_test)

      clf = MultinomialNB(alpha=.01)
      clf.fit(X_train_tfidf, X_train)
      pred = clf.predict(transform_test)
    elif "submit_user" in request.form:
      success = request.form['text']

      text = success
      vectors_test = vectorizer.transform([text,])
      transform_test = tfidf_transformer.transform(vectors_test)
      clf = MultinomialNB(alpha=.01)
      clf.fit(X_train_tfidf, X_train)
      pred = clf.predict(transform_test)
    
  return render_template('dashboard.html', pred = pred)

@app.route('/dashboard')
def login():
  return render_template('dashboard.html')

@app.route('/dataset')
def dataset():
  return render_template('dataset.html', main = csv_dataset, test = csv_test,  train = csv_train, pred = csv_prediction, classification = csv_metrics_dataframe )

@app.route('/categories')
def categories():

  return render_template('categories.html' )

@app.route('/statistics')
def statistics():
  len_dataset = csv_dataset.shape
  len_train = Y_train.shape
  len_test = X_test.shape
  accuracy = float("%0.2f" % (metrics.accuracy_score(X_test, naive)* 100))
  fi_score = float("%0.2f" % (metrics.f1_score(X_test, naive, average='macro')))
  time = float("%0.2f" % (stop - start))
  precision = float("%0.2f" % (metrics.precision_score(X_test, naive, average='macro')))
  recall = float("%0.2f" % (metrics.recall_score(X_test, naive, average='macro')))
  confusion = metrics.confusion_matrix(X_test, naive)

  return render_template('statistics.html', len = len_dataset, lentrain = len_train,  lentest = len_test, accuracy = accuracy , fi_score = fi_score, duration = time, precision = precision, recall = recall )

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
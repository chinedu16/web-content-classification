from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello World Chinedu'

@app.route('/hello')
def new_route():
  return 'testing new routes'

@app.route('/dashboard')
def dashboard():
  return render_template('dashboard.html')

@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username

@app.route('/home')
@app.route('/home/<username>')
def home(name=None):
  return render_template('home.html', name=name)
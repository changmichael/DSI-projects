# This script runs the application on a local server.
# It contains the definition of routes and views for the application.

import flask
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

#---------- MODEL IN MEMORY ----------------#

# Read in the titanic data and build a model on it
df = pd.read_csv('data/speed_dating_user_attributes.csv')
include = ['subject_id', 'wave', 'like_sports', 'like_tvsports',
       'like_exercise', 'like_food', 'like_museums', 'like_art',
       'like_hiking', 'like_gaming', 'like_clubbing', 'like_reading',
       'like_tv', 'like_theater', 'like_movies', 'like_concerts',
       'like_music', 'like_shopping', 'like_yoga',
       'subjective_attractiveness', 'subjective_sincerity',
       'subjective_intelligence', 'subjective_fun', 'subjective_ambition']

# Create dummies and drop NaN
# df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
df = df[include].dropna()

X = df[['like_sports', 'like_tvsports',
       'like_exercise', 'like_food', 'like_museums', 'like_art',
       'like_hiking', 'like_gaming', 'like_clubbing', 'like_reading',
       'like_tv', 'like_theater', 'like_movies', 'like_concerts',
       'like_music', 'like_shopping', 'like_yoga',
       'subjective_sincerity',
       'subjective_fun', 'subjective_ambition']]
y = df['subjective_attractiveness']

PREDICTOR = GradientBoostingRegressor().fit(X, y)





#---------- CREATING AN API, METHOD 1 ----------------#

# Initialize the app
app = flask.Flask(__name__)


# When you navigate to the page 'server/predict', this will run
# the predict() function on the parameters in the url.
#
# Example URL:
# http://localhost:4000/predict?pclass=1&sex=1&age=18&fare=500&sibsp=0
@app.route('/predict', methods=["GET"])
def predict():
    '''Makes a prediction'''
    like_sports = float(flask.request.args['like_sports'])
    like_tvsports = float(flask.request.args['like_tvsports'])
    like_exercise = float(flask.request.args['like_exercise'])
    like_food = float(flask.request.args['like_food'])
    like_museums = float(flask.request.args['like_museums'])
    like_art = float(flask.request.args['like_art'])
    like_hiking = float(flask.request.args['like_hiking'])
    like_gaming = float(flask.request.args['like_gaming'])
    like_clubbing = float(flask.request.args['like_clubbing'])
    like_reading = float(flask.request.args['like_reading'])
    like_tv = float(flask.request.args['like_tv'])
    like_theater = float(flask.request.args['like_theater'])
    like_movies = float(flask.request.args['like_movies'])
    like_concerts = float(flask.request.args['like_concerts'])
    like_music = float(flask.request.args['like_music'])
    like_shopping = float(flask.request.args['like_shopping'])
    like_yoga = float(flask.request.args['like_yoga'])
    subjective_sincerity = float(flask.request.args['subjective_sincerity'])
    subjective_fun = float(flask.request.args['subjective_fun'])
    subjective_ambition = float(flask.request.args['subjective_ambition'])


    item = np.array([like_sports, like_tvsports,
       like_exercise, like_food, like_museums, like_art,
       like_hiking, like_gaming, like_clubbing, like_reading,
       like_tv, like_theater, like_movies, like_concerts,
       like_music, like_shopping, like_yoga,
       subjective_sincerity,
       subjective_fun, subjective_ambition])
    score = PREDICTOR.predict_proba(item)
    results = {'subject attractiveness': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)



#---------- CREATING AN API, METHOD 2 ----------------#


# This method takes input via an HTML page
@app.route('/page')
def page():
   with open("page.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       pclass = inputs['pclass'][0]
       sex = inputs['sex'][0]
       age = inputs['age'][0]
       fare = inputs['fare'][0]
       sibsp = inputs['sibsp'][0]

       item = np.array([pclass, sex, age, fare, sibsp])
       score = PREDICTOR.predict_proba(item)
       results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       return flask.jsonify(results)


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'

    app.run(HOST, PORT)
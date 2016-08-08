# This script runs the application on a local server.
# It contains the definition of routes and views for the application.

import flask
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

#---------- MODEL IN MEMORY ----------------#

# Read in the titanic data and build a model on it
# df = pd.read_csv('data/titanic.csv')
# include = ['day', 'month', 'hour', 'ent', 'exit']

# X = df[['day', 'month', 'hour']]
# y1 = df['ent']
# y2 = df['exit']

# PREDICTOR1 = LinearRegression().fit(X, y1)
# PREDICTOR2 = LinearRegression().fit(X, y2)


#---------- CREATING AN API, METHOD 1 ----------------#
 
# Initialize the app
app = flask.Flask(__name__)

with open('entries.pkl', 'r') as picklefile:
    PREDICTOR1 = pickle.load(picklefile)

with open('exits.pkl', 'r') as picklefile:
    PREDICTOR2 = pickle.load(picklefile)


# When you navigate to the page 'server/predict', this will run
# the predict() function on the parameters in the url.
#
# Example URL:
# http://localhost:4000/predict?pclass=1&sex=1&age=18&fare=500&sibsp=0
@app.route('/predict', methods=["GET"])
def predict():
    '''Makes a prediction'''

    result1 = []
    result2 = []
    day = float(flask.request.args['day'])
    month = float(flask.request.args['month'])
    for i in range(4,25,4):
        hour = i
        item = np.array([day,month,hour])
        # item = np.array([1,2,hour])
        score1 = PREDICTOR1.predict(item)[0]
        score2 = PREDICTOR2.predict(item)[0]
        result1.append(score1)
        result2.append(score2)
    
    # station = float(flask.request.args['station'])


    # item = np.array([day,month,hour])
    # score1 = PREDICTOR1.predict_proba(item)
    # score2 = PREDICTOR2.predict_proba(item)
    # score = PREDICTOR.predict_proba(item)
    # results = {'survival chances': score[0,1], 'death chances': score[0,0]}

    # results = {"in":{"0":1000,"4":1500,"8":1000,"12":1000,"16":1000,"20":1000,"24":1000},"out":{"0":1000,"4":1500,"8":1000,"12":1000,"16":1000,"20":1000,"24":1000}}
    # results = {"in":{,1500,1000,1000,1000,1000,1000},"out":{2000,2500,3000,2000,2000,1200,1200}}
    results = {"in":result1,"out":result2}
    

    return flask.jsonify(results)



#---------- CREATING AN API, METHOD 2 ----------------#


# This method takes input via an HTML page
# @app.route('/page')
# def page():
#    with open("page.html", 'r') as viz_file:
#        return viz_file.read()
#
# @app.route('/result', methods=['POST', 'GET'])
# def result():
#     '''Gets prediction using the HTML form'''
#     if flask.request.method == 'POST':
#
#        inputs = flask.request.form
#
#        pclass = inputs['pclass'][0]
#        sex = inputs['sex'][0]
#        age = inputs['age'][0]
#        fare = inputs['fare'][0]
#        sibsp = inputs['sibsp'][0]
#
#        item = np.array([pclass, sex, age, fare, sibsp])
#        score = PREDICTOR.predict_proba(item)
#        results = {'survival chances': score[0,1], 'death chances': score[0,0]}
#        return flask.jsonify(results)


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'

    app.run(HOST, PORT)

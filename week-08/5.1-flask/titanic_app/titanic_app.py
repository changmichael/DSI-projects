import flask
app = flask.Flask(__name__)

# ----- MODEL GOES HERE ----------#
import numpy as numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/titanic.csv')
include = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']

# Create dummies and drop NaNs
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
df = df[include].dropna()

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
y = df['Survived']

PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y) 

# ----- ROUTES GO HERE -----------#

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
	HOST = '127.0.0.1'
	PORT = '4000'

	app.run(HOST,PORT)


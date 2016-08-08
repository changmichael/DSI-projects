import flask
app = flask.Flask(__name__)

@app.route("/")
def hello():
	return "Hello World!"

@app.route("/greet/<name>")
def greet(name):
	return "<h2>Hello, %s!</h2>" %name

if __name__ == '__main__':
	app.run()


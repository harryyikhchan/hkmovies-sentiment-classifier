import os
from flask import Flask, current_app, request, jsonify
from flask import render_template
from gevent.wsgi import WSGIServer
import pandas
import os

from intent_classifier import IntentClassifier

## Train Data Set File
DATA_PATH = "./data/train_set.csv"


app = Flask(__name__)

# Read the Train Set
data = pandas.read_csv(DATA_PATH, encoding='utf-8',
                    names=['intent', 'utterance'], header=1)

intent_classifier = IntentClassifier()
X = data['utterance']
y = data['intent']
# Train the Engine
intent_classifier.fit(X, y, cv=None)

@app.route('/predict', methods=['POST'])
def predict():
	input = request.json['input']
	results = intent_classifier.predict(input)
	return jsonify(results)

if __name__ == '__main__':
	server = WSGIServer(('', int(os.environ.get('PORT', 5000))), app)
	server.serve_forever()
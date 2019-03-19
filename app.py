from flask import Flask
from flask import jsonify
from flask import make_response
from urllib.error import HTTPError,URLError
from flask import request
import requests
from textblob import TextBlob
import re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

def clean_tweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

global Classifier
global Vectorizer

from rake_nltk import Rake

# load data
data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)



app = Flask(__name__)
@app.route('/tag', methods=['POST'])
def text():
    try:
        data = request.get_json()

        r = Rake()

        r.extract_keywords_from_text(data["text"])



        return make_response(jsonify({"tags": r.get_ranked_phrases()}), 200)

    except HTTPError as e:
        print(e.code)
        return str(e) + 'HTTPError'
    except URLError as e:
        print(e.args)
        return str(e) + 'Url Error'




@app.route('/sentiment', methods=['POST'])
def senti():
    try:
        data = request.get_json()
        blob = TextBlob(clean_tweet(data["text"]))
        for sentence in blob.sentences:
            sent = sentence.sentiment.polarity

        return make_response(jsonify({"sentiment": sent}), 200)

    except HTTPError as e:
        print(e.code)
        return str(e) + 'HTTPError'
    except URLError as e:
        print(e.args)
        return str(e) + 'Url Error'



@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json()
    message = message["text"]
    error = ''
    predict = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message) > 0:
          vectorize_message = Vectorizer.transform([message])
          predict = Classifier.predict(vectorize_message)[0]
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(
              predict=predict)

@app.route('/train', methods=['GET'])
def train_spam():
    data = pandas.read_csv('spam.csv', encoding='latin-1')
    train_data = data[:4400] # 4400 items
    test_data = data[4400:] # 1172 items


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port= '8800',debug=True)

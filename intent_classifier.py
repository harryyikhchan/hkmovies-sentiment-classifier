import jieba
import codecs
import pandas
import re
import numpy as np
import time
import json
import os
import json
import sys 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score


# Global Variable
result = {}

class IntentClassifier:
    
    def __init__(self):
        self.env = os.getenv("ENV", "development")

    def _isChinese(self, s):
        return bool(re.compile('[^\x00-\x7F]+', re.UNICODE).search(s))

    def _preprocessing(self, sentence):
        # Remove Space between Chinese Character
        sentence = re.sub(r"(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F])", '', sentence)
        # Remove Extra Space
        sentence = re.sub(r"^\s+,\s+$", '', sentence)
        sentence = re.sub(r"\s+", ' ', sentence)
            
        return sentence


    def _tokenize(self, sentence):
        sentence = sentence.strip()
        original_sentence = sentence
        print(sentence)
        sentence = self._preprocessing(sentence)

        ## Option 1: Word-Level Tokenizer
        tokens = jieba.lcut(sentence, HMM=False)

        ## Option 2: Character-Level Tokenizer
        # tokens = [t for t in sentence]

        if not tokens:
            tokens = [original_sentence]
        # if (self.env == "development"):
        print(original_sentence, ":", "/".join(tokens))

        return tokens

    def fit(self, X, y, cv=5):

        ## Option 1: TFIDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize, norm=None, min_df=0.0005, max_df=0.9)
        
        ## Option 2: Count Vectorizer
        # self.vectorizer = CountVectorizer(tokenizer=self._tokenize)
        
        ## With LSA 
        ## Remarks: Amend the Number of Vocabs to lower value, e.g 2500 for Character-Level Tokenizer
        svd = TruncatedSVD(4000)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        classifier = Pipeline([
            # ('vect', self.vectorizer),
            ('tfidf', self.vectorizer),
            # ('lsa', lsa),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight="balanced"))
        ])

        classifier.fit(X, y)

        print("Number of vocab = ", len(self.vectorizer.vocabulary_))
        print(sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1]))

        if cv:
            scores = cross_val_score(classifier, X, y, cv=5)
            print(scores)
            print("Accuracy: %0.2f (+/- %0.2f)" %
                  (scores.mean(), scores.std() * 2))
        self.classifier = classifier

        print("Training completed.")

    def predict(self, input):
        # Initialization
        result["query"] = input
        result["intents"] = []
        import csv

        proba = self.classifier.predict_proba([input])[0]

        for cls, prob in zip(self.classifier.classes_, proba):
            prediction = {"intent": cls, "score": prob}
            result["intents"].append(prediction)
        # Sort the result by score
        result["intents"].sort(key=lambda item: item["score"], reverse=True)
        intentWithMaxScore = max(
            result["intents"], key=lambda item: item["score"])
        result["intent"] = intentWithMaxScore["intent"]
        result["score"] = intentWithMaxScore["score"]
        print(result["intents"])
        print(result["intent"], result["score"])

        return result

    def eval(self, X, y):
        import csv

        f = codecs.open("wrong_log.csv", "w", "utf-8")
        writer = csv.writer(f)

        y_pred = self.classifier.predict(X)
        print('Accuracy (Scale: 0-1): ', accuracy_score(y, y_pred))

        for i in range(len(y)):
            if y[i] != y_pred[i]:
                # print y[i], X[i]
                writer.writerow(
                    [y[i], X[i], y_pred[i]])

        f.close()

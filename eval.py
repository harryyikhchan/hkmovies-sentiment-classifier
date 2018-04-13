import pandas

from intent_classifier import IntentClassifier

DATA_PATH = "./data/train_set.csv"
TEST_DATA_PATH = "./data/test_set.csv"

data = pandas.read_csv(DATA_PATH, encoding='utf-8',
                    #    names=['intent', 'utterance', 'instance_id', 'date', 'remarks'], header=0)
                    names=['intent', 'utterance'], header=1)

test_data = pandas.read_csv(TEST_DATA_PATH, encoding='utf-8',
                            names=['intent', 'utterance'], header=0)

intent_classifier = IntentClassifier()
X = data['utterance']
y = data['intent']
intent_classifier.fit(X, y, cv=None)

X_test = test_data['utterance']
y_test = test_data['intent']
intent_classifier.eval(X_test, y_test)

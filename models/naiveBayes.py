import numpy as np
from sklearn import svm

class naiveBayes():
    def __init__(self):
        self.gnb = GaussianNB()

    def preprocess_inputs(self, inputs):

        # list of tuples where the tuple is a feature vector
        inputs = [
            (
                int(x['id']),
                len(x['targetKeywords'].split(','))
            )

            for x in inputs
        ]

        return inputs

    def fit(self, X, y):
        self.gnb.fit(X, y)

    def predict(self, X):
        return self.gnb.predict(X)

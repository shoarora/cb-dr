import numpy as np
from sklearn import svm

class SVM():
    def __init__(self):
        # support vector regression
        self.svr = svm.SVR()
        # support vector classification
        self.svc = svm.SVC()

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
        self.svr.fit(X, y)
        #self.svc.fit(X, y)

    def predict(self, X):
        return self.svr.predict(X)

from sklearn import svm
from feature_extraction import basic_feature_extraction


class SVM():
    def __init__(self):
        # support vector regression
        self.svr = svm.SVR()

        self.num_epochs = 1
        self.batch_size = 25

    def preprocess_inputs(self, inputs):

        # list of tuples where the tuple is a feature vector

        inputs = basic_feature_extraction(inputs)

        return inputs

    def fit(self, X, y):
        self.svr.fit(X, y)

    def predict(self, X):
        return self.svr.predict(X)

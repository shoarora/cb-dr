from sklearn.naive_bayes import GaussianNB
from feature_extraction import basic_feature_extraction


class naiveBayes():
    def __init__(self):
        self.gnb = GaussianNB()

        self.num_epochs = 1
        self.batch_size = 25

    def test(self):
        print 'hello'

    def preprocess_inputs(self, inputs):
        return basic_feature_extraction(inputs)

    def fit(self, X, y):
        self.gnb.fit(X, y)

    def predict(self, X):
        return self.gnb.predict(X)

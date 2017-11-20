from feature_extraction import tfidf_features


class SKBase(object):
    def __init__(self):
        pass

    def preprocess_inputs(self, inputs):
        inputs = tfidf_features(inputs)
        return inputs

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

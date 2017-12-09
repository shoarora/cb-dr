from feature_extraction import tfidf_features


class SKBase(object):
    def __init__(self, choice, freq_floor):
        self.choice = choice
        self.freq_floor = freq_floor

    def preprocess_inputs(self, inputs, ids, path):
        inputs = tfidf_features(path, ids, self.choice, self.freq_floor)
        return inputs

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

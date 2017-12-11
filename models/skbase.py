from feature_extraction import tfidf_features, top_60_feature_extraction


class SKBase(object):
    def __init__(self, choice=None, freq_floor=None):
        self.choice = choice
        self.freq_floor = freq_floor

    def preprocess_inputs(self, inputs, ids=None, path=None):
        #inputs = tfidf_features(path, ids, self.choice, self.freq_floor)
        inputs = top_60_feature_extraction(inputs)
        return inputs

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

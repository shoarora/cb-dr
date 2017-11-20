from feature_extraction import basic_feature_extraction


class SKBase:
    def __init__(self):
        pass

    def preprocess_inputs(self, inputs):
        inputs = basic_feature_extraction(inputs)
        return inputs

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

import numpy as np
from sklearn.kernel_ridge import KernelRidge

class KernelRidge():
    def __init__(self):
        # kernel ridge regression model
        self.model = KernelRidge()

        self.num_epochs = 1
        self.batch_size = 25

    def preprocess_inputs(self, inputs):

        # list of tuples where the tuple is a feature vector
        new_inputs = [
            np.array([
                ('fox' in x['postText'][0].lower()),
                ('fake' in x['postText'][0].lower())
            ], dtype=np.float32)
            for x in inputs
        ]
        return new_inputs

    def fit(self, X, y):
        self.model.fit(X, y)
        #self.svc.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

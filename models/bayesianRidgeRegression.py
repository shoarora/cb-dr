from sklearn import linear_model
from skbase import SKBase


class BayesianRidge(SKBase):
    def __init__(self):
        super(BayesianRidge, self).__init__()
        # bayesian ridge regression model
        self.model = linear_model.BayesianRidge()

        self.num_epochs = 1
        self.batch_size = 25
<<<<<<< Updated upstream
=======

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

    def predict(self, X):
        return self.model.predict(X)
>>>>>>> Stashed changes

from sklearn import linear_model
from skbase import SKBase


class BayesianRidge(SKBase):
    def __init__(self):
        super(BayesianRidge, self).__init__()
        # bayesian ridge regression model
        self.model = linear_model.BayesianRidge()

        self.num_epochs = 1
        self.batch_size = 25

from sklearn.linear_model import LogisticRegression as LR
from skbase import SKBase

class LogisticRegression(SKBase):
    def __init__(self, choice=None, freq_floor=None):
        super(LogisticRegression, self).__init__()
        # support vector regression
        self.model = LR()

        self.num_epochs = 1
        self.batch_size = 25

    def fit(self, X, y):
        y = [1 if i > 0.5 else 0 for i in y]
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.decision_function(X)

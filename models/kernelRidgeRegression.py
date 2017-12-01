from sklearn.kernel_ridge import KernelRidge
from skbase import SKBase
from feature_extraction import basic_feature_extraction

class KernelRidgeRegression(SKBase):
    def __init__(self):
        super(KernelRidgeRegression, self).__init__()
        # kernel ridge regression model
        self.model = KernelRidge()

        self.num_epochs = 1
        self.batch_size = 25

    def preprocess_inputs(self, inputs):
        return basic_feature_extraction(inputs)

    def fit(self, X, y):
        self.model.fit(X, y)
        #self.svc.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

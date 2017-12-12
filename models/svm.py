from sklearn import svm
from skbase import SKBase
from sklearn.preprocessing import MinMaxScaler
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

class SVM(SKBase):
    def __init__(self, choice=None, freq_floor=None):
        super(SVM, self).__init__(choice, freq_floor)
        # support vector regression
        self.model = svm.SVR(kernel='linear')

        self.num_epochs = 1
        self.batch_size = 25

    def fit(self, X, y):
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
        X = scaling.transform(X)
        self.model.fit(X, y)

    def predict(self, X, y):
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
        X = scaling.transform(X)
        return self.model.predict(X)

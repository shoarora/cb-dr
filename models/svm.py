from sklearn import svm
from skbase import SKBase


class SVM(SKBase):
    def __init__(self, choice=None, freq_floor=None):
        super(SVM, self).__init__(choice, freq_floor)
        # support vector regression
        self.model = svm.SVR(kernel='linear')

        self.num_epochs = 1
        self.batch_size = 25

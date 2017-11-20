from sklearn import svm
from skbase import SKBase


class SVM(SKBase):
    def __init__(self):
        super(SVM, self).__init__()
        # support vector regression
        self.model = svm.SVR()

        self.num_epochs = 1
        self.batch_size = 25

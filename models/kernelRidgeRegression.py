from sklearn.kernel_ridge import KernelRidge
from skbase import SKBase


class KernelRidgeRegression(SKBase):
    def __init__(self):
        super(KernelRidgeRegression, self).__init__()
        # kernel ridge regression model
        self.model = KernelRidge()

        self.num_epochs = 1
        self.batch_size = 25

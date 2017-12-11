from sklearn.linear_model import LogisticRegression as LR
from skbase import SKBase

class LogisticRegression(SKBase):
    def __init__(self, choice=None, freq_floor=None):
        super(LogisticRegression, self).__init__()
        # support vector regression
        self.model = LR()

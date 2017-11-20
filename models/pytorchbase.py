import numpy as np
from feature_extraction import tfidf_features
from torch import nn


class TorchBase(nn.Module):
    def __init__(self):
        super(TorchBase, self).__init__()
        self.needs_sess = True
        self.num_epochs = 10
        self.batch_size = 25

    def preprocess_inputs(self, inputs, ids, path):
        return np.array(tfidf_features(path, ids))

    def forward(x):
        raise NotImplemented

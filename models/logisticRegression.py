import numpy as np
import torch.nn as nn
from feature_extraction import tfidf_features

INPUT_DIM = 6
OUTPUT_DIM = 1


class LogisticRegression(nn.Module):
    def __init__(self, input_dimension=INPUT_DIM, output_dimension=OUTPUT_DIM):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(input_dimension, output_dimension)
        self.sigmoid = nn.Sigmoid()

        self.needs_sess = True
        self.num_epochs = 10
        self.batch_size = 25

    def preprocess_inputs(self, inputs, ids, path):

        # list of np.arrays where the tuple is a feature vector
        # return map(np.array, basic_feature_extraction(inputs))
        return np.array(tfidf_features(path, ids))

    def forward(self, inputs):
        out = self.logistic(inputs)
        return self.sigmoid(out)

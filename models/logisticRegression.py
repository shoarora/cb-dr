import numpy as np
import torch.nn as nn

INPUT_DIM = 2
OUTPUT_DIM = 1


class LogisticRegression(nn.Module):
    def __init__(self, input_dimension=INPUT_DIM, output_dimension=OUTPUT_DIM):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(input_dimension, output_dimension)
        self.sigmoid = nn.Sigmoid()

        self.needs_sess = True
        self.num_epochs = 10
        self.batch_size = 25

    def preprocess_inputs(self, inputs):

        # list of np.arrays where the tuple is a feature vector
        new_inputs = [
            np.array([
                ('fox' in x['postText'][0].lower()),
                ('fake' in x['postText'][0].lower())
            ], dtype=np.float32)
            for x in inputs
        ]
        return new_inputs

    def forward(self, inputs):
        out = self.logistic(inputs)
        return self.sigmoid(out)

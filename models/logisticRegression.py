import numpy as np
import torch.nn as nn

INPUT_DIM = 2
OUTPUT_DIM = 1

class LogisticRegression(nn.Module):
    def __init__(self, input_dimension=INPUT_DIM, output_dimension=OUTPUT_DIM):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(input_dimension, output_dimension)

    def preprocess_inputs(self, inputs):

        # list of tuples where the tuple is a feature vector
        inputs = [
            (
                int(x['id']),
                len(x['targetKeywords'].split(','))
            )

            for x in inputs
        ]

        return inputs

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def forward(self, inputs):
        out = self.logistic(inputs)
        return sigmoid(out)

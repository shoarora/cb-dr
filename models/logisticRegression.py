from pytorchbase import TorchBase
from torch import nn

INPUT_DIM = 9720
OUTPUT_DIM = 1


class LogisticRegression(TorchBase):
    def __init__(self, input_dimension=INPUT_DIM, output_dimension=OUTPUT_DIM):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(input_dimension, output_dimension)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out = self.logistic(inputs)
        return self.sigmoid(out)

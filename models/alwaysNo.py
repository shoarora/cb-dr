import numpy as np
import torch.nn as nn


class AlwaysNo(nn.Module):
    def __init__(self):
        super(AlwaysNo, self).__init__()
        self.needs_sess = False

        self.num_epochs = 1
        self.batch_size = 25

    def forward(self, inputs):
        return inputs

    def preprocess_inputs(self, inputs):
        return [0.0] * len(inputs)

import numpy as np
import torch.nn as nn


class AlwaysNo(nn.Module):
    def __init__(self):
        super(AlwaysNo, self).__init__()
        self.needs_sess = False

    def forward(self, inputs):
        return inputs

    def preprocess_inputs(self, inputs):
        return [0.0] * len(inputs)

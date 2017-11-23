import torch
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids

INPUT_DIM = 300
OUTPUT_DIMS = [256, 512, 64, 1]


class VanillaNN(TorchBase):

    def __init__(self):
        super(VanillaNN, self).__init__()
        self.num_words = 100
        self.load_glove()
        self.num_epochs = 100
        dims = [INPUT_DIM] + OUTPUT_DIMS
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                     for i in range(len(dims)-1)])
        self.activation = nn.Sigmoid()

    def preprocess_inputs(self, inputs, ids, path):
        new_inputs = get_word_ids(inputs, self.vocab, self.num_words)
        return new_inputs

    def forward(self, x):
        x = self.embedding(x.long())  # [batch x num_words x glove]
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return torch.mean(x, 1)

import torch
import torch.nn.functional as F
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids

INPUT_DIM = 300
KERNEL_SIZES = [3, 4, 5]
KERNEL_DIM = 100


class CNN(TorchBase):

    def __init__(self):
        super(CNN, self).__init__()
        self.num_words = 100
        self.dropout_p = 0.2
        self.dropout = nn.Dropout(self.dropout_p)
        self.load_glove()
        self.convs = nn.ModuleList([nn.Conv2d(1, KERNEL_DIM, (K, INPUT_DIM))
                                    for K in KERNEL_SIZES])
        self.linear = nn.Linear(None, 1)  # TODO what dim?

    def preprocess_inputs(self, inputs, ids, path):
        new_inputs = get_word_ids(inputs, self.vocab, self.num_words)
        return new_inputs

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (B,1,T,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        concated = torch.cat(x, 1)
        concated = self.dropout(concated)

        return self.linear(x)

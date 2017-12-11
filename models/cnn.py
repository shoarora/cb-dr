import torch
import torch.nn.functional as F
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids

INPUT_DIM = 300
KERNEL_SIZES = [3, 4, 5]
KERNEL_DIM = 100


class CNN(TorchBase):

    def __init__(self, input_dim=INPUT_DIM, kernel_sizes=KERNEL_SIZES,
                 kernel_dim=KERNEL_DIM, load_glove=True, choice='text'):
        super(CNN, self).__init__()
        self.num_words = 100
        self.dropout_p = 0.2
        self.dropout = nn.Dropout(self.dropout_p)
        self.choice = choice

        if load_glove:
            self.load_glove()

        self.input_dim = input_dim
        self.kernel_sizes = kernel_sizes
        self.kernel_dim = kernel_dim

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, input_dim))
                                    for K in kernel_sizes])
        self.fc_dim = 300
        self.linear = nn.Linear(self.fc_dim, 1)  # TODO what dim?

    def preprocess_inputs(self, inputs, ids, path):
        new_inputs = get_word_ids(inputs, self.vocab, self.num_words, target=self.choice)
        return new_inputs

    def extract_features(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        concated = torch.cat(x, 1)
        concated = self.dropout(concated)
        return concated

    def forward(self, x):
        x = self.embedding(x.long())  # (B,1,T,D)
        x = self.extract_features(x)
        return self.linear(x)

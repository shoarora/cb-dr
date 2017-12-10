import numpy as np
import torch
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids
from rnn import RNN

INPUT_DIM = 300
HIDDEN_SIZE = 256

# KERNEL_DIM = 100
# KERNEL_SIZES = [3, 4, 5]


class ParallelNet(TorchBase):

    def __init__(self):
        super(ParallelNet, self).__init__()
        self.num_words = 300
        self.glove_dim = INPUT_DIM
        self.load_glove()
        self.num_epochs = 100

        self.net1 = RNN(input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                        load_glove=False)
        self.net2 = RNN(input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                        load_glove=False)

        self.net1.max_len = self.num_words
        self.net1.glove_dim = INPUT_DIM
        self.net2.max_len = self.num_words
        self.net2.glove_dim = INPUT_DIM

        self.combine = nn.Linear(HIDDEN_SIZE * 2 * self.net1.n_layers,
                                 HIDDEN_SIZE * self.net1.n_layers)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(HIDDEN_SIZE * 3 * self.net1.n_layers, 1)

    def preprocess_inputs(self, inputs, ids, path):
        post_inputs = get_word_ids(inputs, self.vocab,
                                   self.num_words, target='post')
        title_inputs = get_word_ids(inputs, self.vocab,
                                    self.num_words, target='title')
        new_inputs = [np.array(x, dtype=np.int32)
                      for x in zip(post_inputs, title_inputs)]
        return new_inputs

    def forward(self, x):
        x, y = torch.chunk(x, 2, dim=1)
        x = torch.squeeze(x)
        y = torch.squeeze(x)

        x = self.embedding(x.long())
        y = self.embedding(y.long())

        x = self.net1.extract_features(x)
        y = self.net2.extract_features(y)

        z = self.combine(torch.cat([x, y], 1))
        z = self.activation(z)
        return self.fc(torch.cat([x, y, z], 1))

import numpy as np
import torch
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids
from rnn import RNN
from cnn import CNN

INPUT_DIM = 300
HIDDEN_SIZE = 256

# KERNEL_DIM = 100
# KERNEL_SIZES = [3, 4, 5]


class ParallelNet3(TorchBase):

    def __init__(self):
        super(ParallelNet, self).__init__()
        self.num_words = 100
        self.glove_dim = INPUT_DIM
        self.load_glove()
        self.num_epochs = 100

        self.net1 = self.init_CNN()
        self.net2 = self.init_CNN()
        self.net3 = self.init_CNN()

        self.combine_dim = 256
        self.combine = nn.Linear(self.net1.fc_dim + self.net2.fc_dim + self.net3.fc_dim,
                                 self.combine_dim)
        self.activation = nn.Tanh()
        self.fc_dim = self.net1.fc_dim + self.net2.fc_dim + self.net3.fc_dim + self.combine_dim
        self.fc = nn.Linear(self.fc_dim, 1)

    def init_CNN(self):
        net = CNN(load_glove=False)
        net.num_words = self.num_words
        net.glove_dim = INPUT_DIM
        return net

    def init_RNN(self):
        net = RNN(input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                  load_glove=False)
        net.max_len = self.num_words
        net.glove_dim = INPUT_DIM
        return net

    def preprocess_inputs(self, inputs, ids, path):
        post_inputs = get_word_ids(inputs, self.vocab,
                                   self.num_words, target='post')
        title_inputs = get_word_ids(inputs, self.vocab,
                                    self.num_words, target='title')
        text_inputs = get_word_ids(inputs, self.vocab,
                                   self.num_words, target='text')
        new_inputs = [np.array(x, dtype=np.int32)
                      for x in zip(post_inputs, title_inputs, text_inputs)]
        return new_inputs

    def forward(self, x):
        x, y, z = torch.chunk(x, 3, dim=1)
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        z = torch.squeeze(z)

        x = self.embedding(x.long())
        y = self.embedding(y.long())
        z = self.embedding(z.long())

        x = self.net1.extract_features(x)
        y = self.net2.extract_features(y)
        z = self.net3.extract_features(z)

        out = self.combine(torch.cat([x, y, z], 1))
        out = self.activation(out)
        return self.fc(torch.cat([x, y, z, out], 1))

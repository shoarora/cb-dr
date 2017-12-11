import numpy as np
import torch
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids, top_60_feature_extraction
from rnn import RNN
from cnn import CNN

INPUT_DIM = 300
HIDDEN_SIZE = 256

# KERNEL_DIM = 100
# KERNEL_SIZES = [3, 4, 5]


class ParallelNet3(TorchBase):

    def __init__(self, classify=False, add_top60=False):
        super(ParallelNet3, self).__init__()
        self.num_words = 100
        self.glove_dim = INPUT_DIM
        self.load_glove()
        self.num_epochs = 100
        self.out_dim = 2 if classify else 1
        self.add_top60 = add_top60

        self.net1 = self.init_CNN()
        self.net2 = self.init_CNN()
        self.net3 = self.init_CNN()

        self.combine_dim = 256
        self.combine = nn.Linear(self.net1.fc_dim + self.net2.fc_dim + self.net3.fc_dim,
                                 self.combine_dim)
        self.activation = nn.Tanh()
        self.fc_dim = self.net1.fc_dim + self.net2.fc_dim + self.net3.fc_dim + self.combine_dim
        if add_top60:
            self.fc_dim += 50
        self.fc = nn.Linear(self.fc_dim, self.out_dim)

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
        new_inputs = []
        new_inputs.append(get_word_ids(inputs, self.vocab,
                          self.num_words, target='post'))
        new_inputs.append(get_word_ids(inputs, self.vocab,
                          self.num_words, target='title'))
        new_inputs.append(get_word_ids(inputs, self.vocab,
                          self.num_words, target='text'))
        if self.add_top60:
            new_inputs.append([
                np.array(x) for x in top_60_feature_extraction(inputs)
            ])

        new_inputs = [np.array(x, dtype=np.int32)
                      for x in zip(*new_inputs)]
        return new_inputs

    def forward(self, x):
        if self.add_top60:
            x, y, z, top60 = torch.chunk(x, 4, dim=1)
        else:
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

        if self.add_top60:
            final = torch.cat([x, y, z, out, torch.squeeze(top60)], 1)
        else:
            final = torch.cat([x, y, z, out], 1)

        return self.fc(final)

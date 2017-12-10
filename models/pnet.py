import numpy as np
import torch
from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids
from rnn import RNN

INPUT_DIM = 50
HIDDEN_SIZE = 256

# KERNEL_DIM = 100
# KERNEL_SIZES = [3, 4, 5]


class ParallelNet(TorchBase):

    def __init__(self):
        super(ParallelNet, self).__init__()
        self.num_words = 100
        self.load_glove()
        self.num_epochs = 100
        # TODO init nets
        # TODO init fc
        self.net1 = RNN(input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE)
        self.net2 = RNN(input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE)

        self.combine = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(HIDDEN_SIZE * 3, HIDDEN_SIZE)

    def preprocess_inputs(self, inputs, ids, path):
        post_inputs = get_word_ids(inputs, self.vocab,
                                   self.num_words, target='post')
        title_inputs = get_word_ids(inputs, self.vocab,
                                    self.num_words, target='title')
        # return zip(post_inputs, title_inputs)  # TODO adjust dataloader to accept this OR just pass them as 2d
        new_inputs = [np.array(x, dtype=np.int32) for x in zip(post_inputs, title_inputs)]
        return new_inputs

    def forward(self, x):
        x, y = torch.chunk(x, 2, dim=1)
        x = torch.squeeze(x)
        y = torch.squeeze(x)
        print x.size(), y.size()
        x = self.embedding(x.long())  # [batch x num_words x glove]
        y = self.embedding(y.long())
        print x.size(), y.size()
        x = self.net1.extract_features(x)
        y = self.net2.extract_features(x)
        z = self.combine(torch.concat([x, y], 0))
        z = self.activation(z)
        return self.fc(torch.concat([x, y, z], 0))

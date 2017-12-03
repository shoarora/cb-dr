from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids

INPUT_DIM = 300
HIDDEN_SIZE = 256


class RNN(TorchBase):
    def __init__(self):
        super(RNN, self).__init__()
        self.load_glove()

        # TODO how to deal with variable lengths

        self.max_len = 300
        self.input_dropout_p = 0.2
        self.n_layers = 4
        self.bidirectional = False
        self.variable_lengths = True
        self.rnn = nn.GRU(INPUT_DIM,
                          HIDDEN_SIZE,
                          self.n_layers,
                          batch_first=True,
                          dropout=self.input_dropout_p,
                          bidirectional=self.bidirectional)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)  # TODO what is the actual dim of this

    def preprocess_inputs(self, inputs, ids, path):
        new_inputs = get_word_ids(inputs, self.vocab)
        return new_inputs

    def forward(self, x):
        x = self.embedding(x)
        x, h = self.rnn(x)
        print x.shape, h.shape
        return self.linear(h)

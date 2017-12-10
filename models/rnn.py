from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids

INPUT_DIM = 300
HIDDEN_SIZE = 256


class RNN(TorchBase):
    def __init__(self, input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE, load_glove=True):
        super(RNN, self).__init__()

        if load_glove:
            self.load_glove()

        self.max_len = 300
        self.input_dropout_p = 0.2
        self.n_layers = 4
        self.bidirectional = False

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_dim,
                          hidden_size,
                          self.n_layers,
                          batch_first=True,
                          dropout=self.input_dropout_p,
                          bidirectional=self.bidirectional)
        self.linear = nn.Linear(self.n_layers * hidden_size, 1)

    def preprocess_inputs(self, inputs, ids, path):
        new_inputs = get_word_ids(inputs, self.vocab, self.max_len)
        return new_inputs

    def extract_features(self, x):
        _, h = self.rnn(x)
        h = h.permute(1, 0, 2)
        h = h.contiguous().view(-1, self.n_layers * self.hidden_size)
        return h

    def forward(self, x):
        x = self.embedding(x.long())
        h = self.extract_features(x)
        return self.linear(h)

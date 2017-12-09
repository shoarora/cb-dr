from torch import nn
from pytorchbase import TorchBase
from feature_extraction import get_word_ids

INPUT_DIM = 300
HIDDEN_SIZE = 256


class RNN(TorchBase):
    def __init__(self):
        super(RNN, self).__init__()
        self.load_glove()

        self.max_len = 300
        self.input_dropout_p = 0.2
        self.n_layers = 4
        self.bidirectional = False
        self.rnn = nn.GRU(INPUT_DIM,
                          HIDDEN_SIZE,
                          self.n_layers,
                          batch_first=True,
                          dropout=self.input_dropout_p,
                          bidirectional=self.bidirectional)
        self.linear = nn.Linear(self.n_layers * HIDDEN_SIZE, 1)

    def preprocess_inputs(self, inputs, ids, path):
        new_inputs = get_word_ids(inputs, self.vocab, self.max_len)
        return new_inputs

    def forward(self, x):
        x = self.embedding(x.long())
        _, h = self.rnn(x)
        h = h.permute(1, 0, 2)
        h = h.contiguous().view(-1, self.n_layers * HIDDEN_SIZE)
        return self.linear(h)

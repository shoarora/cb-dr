import numpy as np
import torch
from feature_extraction import tfidf_features, load_glove_vecs
from torch import nn


class TorchBase(nn.Module):
    def __init__(self):
        super(TorchBase, self).__init__()
        self.needs_sess = True
        self.num_epochs = 10
        self.batch_size = 25
        self.glove_dim = 50
        self.glove_path = 'data/glove/glove.6B.'+str(self.glove_dim)+'d.txt'

    def preprocess_inputs(self, inputs, ids, path):
        return np.array(tfidf_features(path, ids))

    def forward(self, x):
        raise NotImplemented

    def load_glove(self):
        self.vocab, self.glove = load_glove_vecs(self.glove_path)
        size = self.glove.size()

        # TODO get pad, unk, start, and end working

        pad = torch.FloatTensor(np.zeros((1, self.glove_dim)))
        # unk = torch.FloatTensor(np.random.rand(1, self.glove_dim))

        # self.glove = torch.cat([pad, unk, self.glove], dim=0)
        self.glove = torch.cat([pad, self.glove], dim=0)

        self.embedding = nn.Embedding(size[0], size[1], padding_idx=0)
        self.embedding.weight = nn.Parameter(self.glove)
        self.embedding.weight.requires_grad = False

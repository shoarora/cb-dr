from alwaysNo import AlwaysNo
from logisticRegression import LogisticRegression
from bayesianRidgeRegression import BayesianRidge
from kernelRidgeRegression import KernelRidgeRegression
from svm import SVM
from nn import VanillaNN
from rnn import RNN
from cnn import CNN
from pnet import ParallelNet
import feature_extraction as FE


model_options = {
    'no': AlwaysNo,
    'lr': LogisticRegression,
    'nn': VanillaNN,
    'rnn': RNN,
    'cnn': CNN,
    'pnet': ParallelNet
}

sk_model_options = {
    'bayesRR': BayesianRidge,
    'kernelRR': KernelRidgeRegression,
    'svm': SVM
}

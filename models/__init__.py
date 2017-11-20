from alwaysNo import AlwaysNo
from logisticRegression import LogisticRegression
from bayesianRidgeRegression import BayesianRidge
from kernelRidgeRegression import KernelRidgeRegression
from svm import SVM
import feature_extraction as FE


model_options = {
    'no': AlwaysNo,
    'lr': LogisticRegression
}

sk_model_options = {
    'bayesRR': BayesianRidge,
    'kernelRR': KernelRidgeRegression,
    'svm': SVM
}

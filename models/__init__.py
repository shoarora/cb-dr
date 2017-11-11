from alwaysNo import AlwaysNo
from logisticRegression import LogisticRegression
from naiveBayes import naiveBayes
from svm import SVM

model_options = {
    'no': AlwaysNo,
    'lr': LogisticRegression
}

sk_model_options = {
    'nb': naiveBayes,
    'svm': SVM
}

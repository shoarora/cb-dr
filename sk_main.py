import argparse
from models import sk_model_options
from data import get_datasets
import os
import numpy as np

data_paths = {
    'small': 'data/cb-small',
    'big': 'data/cb-big'
}

def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--train', action='store_true')
    #parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--dataset', choices={'small', 'big'})
    parser.add_argument('--model', choices=sk_model_options.keys())
    #parser.add_argument('--cuda', action='store_true')
    #parser.add_argument('--sess_name')
    return parser

def train(model, train_set):
    X = np.array(train_set.inputs)
    y = np.array(train_set.labels)

    model.fit(X, y)

def equals(x, y):
    return abs(x - y) < 1e-10

def evaluate(model, test_set):
    X = np.array(test_set.inputs)
    labels = np.array(test_set.labels)
    predictions = model.predict(X)

    true_label = 0
    for y, y_hat in zip(labels, predictions):
        if equals(y, y_hat):
            true_label += 1

    print '%d correct out of %d' % (true_label, len(labels))
    false_label = len(labels) - true_label
    error_rate = (false_label / len(labels)) * 100.
    print 'error rate: %.2f%%' % error_rate

def main():
    parser = get_parser()
    args = parser.parse_args()

    # load model.  model_options defined in models/__init__.py
    model = sk_model_options[args.model]()

    # load data
    data_path = data_paths[args.dataset]
    train_set, dev_set, test_set = get_datasets(model.batch_size, data_path,
                                                model.preprocess_inputs, sk=True)

    train(model, train_set)

    evaluate(model, test_set)

if __name__ == "__main__":
    main()

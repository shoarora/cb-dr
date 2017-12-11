import argparse
import os
from util import write_predictions_to_file, mkdir
from eval import evaluate_results
from models import sk_model_options
from data import get_datasets
import numpy as np

data_paths = {
    'mini': 'data/cb-mini',
    'small': 'data/cb-small',
    'big': 'data/cb-big'
}

CKPT = 'checkpoints'


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--dataset', choices={'mini', 'small', 'big'})
    parser.add_argument('--model', choices=sk_model_options.keys())
    # parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--sess_name')
    parser.add_argument('--choice', choices={'post', 'text', 'title'})
    parser.add_argument('--freq_floor', type=int)
    return parser


def train(model, train_set):
    X = np.array(train_set.inputs)
    y = np.array(train_set.labels)

    model.fit(X, y)


def evaluate(model, test_set, results_dir, name, truth_file):
    ids = np.array(test_set.ids)
    X = np.array(test_set.inputs)
    predictions = model.predict(X)

    results = {}
    for id, output in zip(ids, predictions):
        results[id] = output

    predictions_file = os.path.join(results_dir, name+'_predictions.json')
    output_file = os.path.join(results_dir, name+'_output.prototext')

    write_predictions_to_file(results, predictions_file)
    print '\n' * 2
    accuracy = evaluate_results(truth_file, predictions_file, output_file)

    return accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    # load model.  model_options defined in models/__init__.py
    model = sk_model_options[args.model](args.choice, args.freq_floor)

    # load data
    data_path = data_paths[args.dataset]
    train_set, dev_set, test_set = get_datasets(model.batch_size, data_path,
                                                model.preprocess_inputs, sk=True)

    print 'training...'
    train(model, train_set)
    print 'done training.'

    truth_file = os.path.join(data_path, 'truth.jsonl')
    mkdir(os.path.join(CKPT, args.sess_name))
    results_dir = os.path.join(CKPT, args.sess_name, 'results')
    mkdir(results_dir)
    print 'evaluating...'
    evaluate(model, dev_set, results_dir, 'dev', truth_file)
    evaluate(model, test_set, results_dir, 'test', truth_file)
    print 'done evaluating.'


if __name__ == "__main__":
    main()

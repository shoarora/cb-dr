import argparse
import json
import os
import torch
from torch.autograd import Variable

from data import get_datasets
from eval import evaluate_results
from models import model_options
from util import Progbar


CKPT = 'checkpoints'

# TODO:
# save stopping point in training
# write readme
# store constants/configs in model
# build out models


data_paths = {
    'small': 'cb-small',
    'big': 'cb-big'
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--dataset', choices={'small', 'big'})
    parser.add_argument('--model', choices=model_options.keys())
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--sess_name')
    return parser


def run_epoch(model, epoch, datasets, optimizer, criterion, cuda, results_dir, truth_file):
    train_loader, dev_loader, test_loader = datasets

    print 'Training epoch', epoch+1
    train(model, train_loader, optimizer, criterion, cuda)

    print 'Evaluating dev epoch', epoch+1
    evaluate(model, dev_loader, optimizer, criterion, cuda, results_dir, 'dev'+str(epoch), truth_file)

    save_model = None  # TODO
    return save_model


def train(model, train_loader, optimizer, criterion, cuda):
    prog = Progbar(len(train_loader))
    for j, data in enumerate(train_loader, 1):
        ids, inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        if torch.cuda.is_available() and cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prog.update(j, values=[('avg_loss', loss.data[0])],
                    exact=[('loss', loss.data[0])])


def evaluate(model, loader, criterion, cuda, results_dir, name, truth_file):
    prog = Progbar(len(loader))
    results = {}
    for j, data in enumerate(loader, 1):
        ids, inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        if torch.cuda.is_available() and cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        prog.update(j, values=[('avg_loss', loss.data[0])],
                    exact=[('loss', loss.data[0])])

        for id, output in zip(ids, outputs):
            results[id] = output

    predictions_file = os.path.join(results_dir, name, 'predictions', '.json')
    output_file = os.path.join(results_dir, name, 'output', '.prototext')

    with open(predictions_file, 'w') as f:
        for id, output in results.iteritems():
            f.write(json.dumps({
                'id': id,
                'clickbaitScore': output
            }))

    evaluate_results(truth_file, predictions_file, output_file)
    # TODO call eval function, save results to dir

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # load model.  model_options defined in models/__init__.py
    model = model_options[args.model]()

    cuda = args.cuda
    if torch.cuda.is_available() and cuda:
        model.cuda()

    optimizer = None  # TODO
    criterion = torch.nn.MSELoss()

    # load saved weights if available
    sess_name = args.sess_name
    if sess_name in os.listdir(CKPT):
        model.load_state_dict(torch.load(os.path.join(CKPT,
                                                      sess_name+'.ckpt')))
    elif args.eval_only and model.needs_sess:  # if eval, we need a saved model
        raise

    # load data
    data_path = data_paths[args.dataset]
    datasets = get_datasets(model.batch_size, data_path,
                            model.preprocess_inputs)
    truth_file = os.path.join(data_path, 'truth.jsonl')

    results_dir = os.path.join(CKPT, sess_name, 'results')

    if args.train:
        for i in model.num_epochs:
            save_model = run_epoch(model, i, datasets, optimizer,
                                   criterion, cuda, results_dir, truth_file)

            if save_model:
                torch.save(model.state_dict(),
                           os.path.join(CKPT, sess_name))

        # evaluate on test set
        test_loader = datasets[2]
        evaluate(model, test_loader, criterion, cuda, results_dir, 'test', truth_file)

    if args.eval_only:
        test_loader = datasets[2]
        evaluate(model, test_loader, criterion, cuda, results_dir, 'test', truth_file)

import argparse
import os
import torch
from torch.autograd import Variable

from data import get_datasets
from eval import evaluate_results
from models import model_options, ParallelNet3
from util import Progbar, mkdir, write_predictions_to_file


BORDER = '_' * 80
CKPT = 'checkpoints'
data_paths = {
    'small': 'data/cb-small',
    'big': 'data/cb-big'
}
device_num = 0


def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--dataset', choices={'small', 'big'})
    parser.add_argument('--model', choices=model_options.keys())
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--sess_name')
    parser.add_argument('--gpu_num', type=int)
    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--choice', choices={'post', 'title', 'text'})
    return parser


def run_epoch(model, epoch, datasets, optimizer,
              criterion, cuda, results_dir, truth_file):
    '''
    Run one epoch's training and dev eval routine
    Args:
        model: (nn.Module)       the model we're using to train/eval
        epoch: (int)             current epoch number
        datasets: ([Dataloader]) dataloaders for train, dev, and test sets
        optimizer: (Optimizer)   optimizer object for model
        criterion: (Criterion)   the criteron we're evaluating
        cuda: (bool)             whether to use cuda
        results_dir: (str)       path to store evaluation results
        truth_file: (str)        path to file where labels for data is stored
    Returns:
        dev_acc: (float) the accuracy the model achieved on the dev set
    '''
    train_loader, dev_loader, _ = datasets

    print 'Training epoch', epoch+1
    train(model, train_loader, optimizer, criterion, cuda)

    evaluate(model,
             train_loader,
             criterion,
             cuda,
             results_dir,
             'train'+str(epoch),
             truth_file)

    print 'Evaluating dev epoch', epoch+1
    dev_acc = evaluate(model,
                       dev_loader,
                       criterion,
                       cuda,
                       results_dir,
                       'dev'+str(epoch),
                       truth_file)
    return dev_acc


def train(model, train_loader, optimizer, criterion, cuda):
    '''
    Run one epoch's training routine
    Args:
        model: (nn.Module)         the model we're using to train/eval
        train_loader: (Dataloader) dataloader for train set
        optimizer: (Optimizer)     optimizer object for model
        criterion: (Criterion)     the criteron we're evaluating
        cuda: (bool)               whether to use cuda
    '''
    model.train()
    prog = Progbar(len(train_loader))
    for j, data in enumerate(train_loader, 1):
        ids, inputs, labels = data

        inputs, labels = Variable(inputs.float()), Variable(labels.float())
        if torch.cuda.is_available() and cuda:
            inputs, labels = inputs.cuda(device_num), labels.cuda(device_num)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prog.update(j, values=[('avg_loss', loss.data[0])],
                    exact=[('loss', loss.data[0])])


def evaluate(model, loader, criterion, cuda, results_dir, name, truth_file):
    '''
    Run one epoch's eval routine
    Args:
        model: (nn.Module)     the model we're using to train/eval
        loader: (Dataloader)   dataloader to evaluate on
        criterion: (Criterion) the criteron we're evaluating
        cuda: (bool)           whether to use cuda
        results_dir: (str)     path to store evaluation results
        name: (str)            name for this individual evaluation of model
        truth_file: (str)      path to file where labels for data is stored
    Returns:
        accuracy: (float)      model's accuracy on provided dataset
    '''
    model.eval()
    prog = Progbar(len(loader))
    results = {}
    for j, data in enumerate(loader, 1):
        ids, inputs, labels = data

        inputs, labels = Variable(inputs.float()), Variable(labels.float())
        if torch.cuda.is_available() and cuda:
            inputs, labels = inputs.cuda(device_num), labels.cuda(device_num)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        prog.update(j, values=[('avg_loss', loss.data[0])],
                    exact=[('loss', loss.data[0])])

        for id, output in zip(ids, outputs):
            if cuda:
                output = output.cpu()
            output = float(output.data.numpy()[0])
            results[str(id)] = output

    predictions_file = os.path.join(results_dir, name+'_predictions.json')
    output_file = os.path.join(results_dir, name+'_output.prototext')

    write_predictions_to_file(results, predictions_file)
    print '\n' * 2
    accuracy = evaluate_results(truth_file, predictions_file, output_file)
    return accuracy


def load_checkpoint(sess_name, model):
    '''
    Load checkpoint for given session
    Args:
        sess_name: (str)   the name of the session we want to load model for
        model: (nn.Module) the model to load from checkpoint
    Returns:
        model: (nn.Module)    the model restored from checkpoint
        best_dev_acc: (float) the best accuracy this model achieved on dev set
        epoch: (int)          the epoch we saved this checkpoint at
    '''
    path = os.path.join(CKPT, sess_name+'.ckpt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    best_dev_acc = checkpoint['best_dev_acc']
    epoch = checkpoint['epoch']
    return model, best_dev_acc, epoch


if __name__ == '__main__':
    mkdir(CKPT)
    parser = get_parser()
    args = parser.parse_args()

    if args.gpu_num is not None:
        device_num = args.gpu_num

    # load model.  model_options defined in models/__init__.py
    if args.choice:
        model = model_options[args.model](choice=args.choice)
    else:
        model = model_options[args.model]()
    best_dev_acc = 0.0
    epoch = 0

    # move to cuda
    cuda = args.cuda
    if torch.cuda.is_available() and cuda:
        model.cuda(device_num)

    # init training optimizer and criterion
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, weight_decay=1e-6)

    if args.l1:
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()

    # load saved weights if available
    sess_name = args.sess_name
    if sess_name+'.ckpt' in os.listdir(CKPT):
        model, best_dev_acc, epoch = load_checkpoint(sess_name, model)
        print 'Loaded old checkpoint for ', sess_name, 'at ', epoch
        print 'Previous best_dev_acc', best_dev_acc
    elif args.eval_only and model.needs_sess:  # if eval, we need a saved model
        raise

    # load data
    data_path = data_paths[args.dataset]
    datasets = get_datasets(model.batch_size, data_path,
                            model.preprocess_inputs)

    # set up storage for eval results
    truth_file = os.path.join(data_path, 'truth.jsonl')
    results_dir = os.path.join(CKPT, sess_name, 'results')
    mkdir(os.path.join(CKPT, sess_name))
    mkdir(results_dir)

    if args.train:
        for i in range(epoch, model.num_epochs):
            dev_acc = run_epoch(model, i, datasets, optimizer,
                                criterion, cuda, results_dir, truth_file)
            if dev_acc > best_dev_acc:
                # if model performs best so far, save it
                best_dev_acc = dev_acc
                print 'saving new best dev_acc', dev_acc
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'epoch': i+1,
                    'best_dev_acc': best_dev_acc
                }
                torch.save(checkpoint, os.path.join(CKPT, sess_name+'.ckpt'))
            print BORDER

        # load best model and eval on test set
        model, _, epoch = load_checkpoint(sess_name, model)
        print 'evaluating data sets on model from epoch', epoch
        loaders_names = zip(datasets, ['train', 'dev', 'test'])
        for loader, name in loaders_names:
            print BORDER
            print name
            evaluate(model, loader, criterion, cuda,
                     results_dir, name, truth_file)

    if args.eval_only:
        print 'evaluating data sets on model from epoch', epoch
        loaders_names = zip(datasets, ['train', 'dev', 'test'])
        for loader, name in loaders_names:
            print BORDER
            print name
            evaluate(model, loader, criterion, cuda,
                     results_dir, name, truth_file)

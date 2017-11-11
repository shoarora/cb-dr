import json
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader


class cbDataset(Dataset):
    def __init__(self, ids, inputs, labels):
        self.ids = ids
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.ids[i], self.inputs[i], self.labels[i]


def process_inputs(inputs, labels, start_i, end_i, preprocess_f):
    inputs = inputs[start_i:end_i]
    labels = labels[start_i:end_i]
    ids = np.array([inp['id'] for inp in inputs])
    return ids, preprocess_f(inputs), labels


def get_datasets(batch_size, path, preprocess_f, sk=False):
    inputs = load_instances_json(path)
    labels = load_truth_json(path)

    num_entries = len(inputs)

    train_ids, train_inputs, train_labels = process_inputs(inputs,
                                                           labels,
                                                           0,
                                                           num_entries / 2,
                                                           preprocess_f)

    dev_ids, dev_inputs, dev_labels = process_inputs(inputs,
                                                     labels,
                                                     num_entries / 2,
                                                     num_entries * 3 / 4,
                                                     preprocess_f)

    test_ids, test_inputs, test_labels = process_inputs(inputs,
                                                        labels,
                                                        num_entries * 3 / 4,
                                                        num_entries,
                                                        preprocess_f)

    train = cbDataset(train_ids, train_inputs, train_labels)
    dev = cbDataset(dev_ids, dev_inputs, dev_labels)
    test = cbDataset(test_ids, test_inputs, test_labels)

    if sk:
        return (train, dev, test)

    return (DataLoader(train, batch_size),
            DataLoader(dev, batch_size),
            DataLoader(test, batch_size))


def load_instances_json(path):
    inputs = []
    with open(os.path.join(path, 'instances.jsonl')) as f:
        for line in f:
            entry = json.loads(line)
            inputs.append(entry)
    return inputs


def load_truth_json(path):
    labels = []
    with open(os.path.join(path, 'truth.jsonl')) as f:
        for line in f:
            entry = json.loads(line)
            labels.append(entry['truthMean'])
    return labels

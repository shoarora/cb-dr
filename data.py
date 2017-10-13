import json
import os

from torch.utils.data import Dataset, DataLoader


class cbDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.labels[i]


def get_datasets(batch_size, path, preprocess_f):
    inputs = load_instances_json(path)
    labels = load_truth_json(path)

    inputs = preprocess_f(inputs)

    num_entries = len(inputs)

    train_inputs = inputs[:num_entries / 2]
    train_labels = labels[:num_entries / 2]

    dev_inputs = inputs[num_entries / 2:num_entries * 3 / 4]
    dev_labels = labels[num_entries / 2:num_entries * 3 / 4]

    test_inputs = inputs[num_entries * 3 / 4:]
    test_labels = labels[num_entries * 3 / 4:]

    train = cbDataset(train_inputs, train_labels)
    dev = cbDataset(dev_inputs, dev_labels)
    test = cbDataset(test_inputs, test_labels)

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

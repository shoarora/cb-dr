import json
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader


class cbDataset(Dataset):
    def __init__(self, ids, inputs, labels, use_images=False):
        self.ids = ids
        self.inputs = inputs
        self.labels = labels
        # TODO support images

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.ids[i], self.inputs[i], self.labels[i]


def get_datasets(batch_size, path, preprocess_f, sk=False):
    def process_inputs(inputs, labels, start_i, end_i):
        inputs = inputs[start_i:end_i]
        labels = labels[start_i:end_i]
        ids = np.array([inp['id'] for inp in inputs], dtype=np.int64)
        return ids, preprocess_f(inputs, ids, path), labels

    inputs = load_instances_json(path)
    labels = load_truth_json(path)

    num_entries = len(inputs)

    train_ids, train_inputs, train_labels = process_inputs(inputs,
                                                           labels,
                                                           0,
                                                           num_entries * 3 / 5,
                                                           )

    dev_ids, dev_inputs, dev_labels = process_inputs(inputs,
                                                     labels,
                                                     num_entries * 3 / 5,
                                                     num_entries * 4 / 5,
                                                     )

    test_ids, test_inputs, test_labels = process_inputs(inputs,
                                                        labels,
                                                        num_entries * 4 / 5,
                                                        num_entries,
                                                        )

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


def write_tf_and_df(path):
    '''
    path to data dir
    number of times word appears in document / number of documents word appears
    tf [(id, word): count]
    idf [word: count]
    '''
    words_to_ids = {}
    tf = {}
    df = {}
    ids = []
    inputs = load_instances_json(path)

    for inp in inputs:
        id = inp['id']
        ids.append(id)
        counts = {}
        for line in inp['postText']:
            for w in line.lower().split(' '):
                # count occurrences of each word within a document
                counts[w] = counts.get(w, 0) + 1
                if w not in words_to_ids:
                    words_to_ids[w] = set([])
                words_to_ids[w].add(id)
        tf[id] = counts

    # count how many documents a word appears in
    for w, df_ids in words_to_ids.iteritems():
        df[w] = len(df_ids)
    results = {
        'ids': ids,
        'term freqs': tf,
        'doc freqs': df
    }
    with open(os.path.join(path, 'frequencies_text.json'), 'w') as f:
        f.write(json.dumps(results))

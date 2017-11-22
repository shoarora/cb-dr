import json
import os
import math
import torchwordemb


def get_word_ids(inputs, vocab, num_words=None):
    new_inputs = []
    for inp in inputs:
        inp = ''.join(inp['targetParagraphs'])
        if num_words:
            inp = inp[:num_words]
        # we reserve index 0 for padding
        # TODO actually figure out what to do with unkown keys
        new_input = [vocab.get(word, 0)+1 for word in inp.lower().split(' ')]
        new_inputs.append(new_input)
    return new_inputs


def load_glove_vecs(path):
    vocab, vec = torchwordemb.load_glove_text(path)
    return vocab, vec


def average_paragraph_length(targetParagraphs):
    sum = 0.0
    if len(targetParagraphs) == 0:
        return 0
    for i in range(0, len(targetParagraphs)):
        sum += len(targetParagraphs[i])
    return sum / len(targetParagraphs)


def average_word_length(targetParagraphs):
    sum = 0.0
    total = 0
    if len(targetParagraphs) == 0:
        return 0, 0
    for i in range(0, len(targetParagraphs)):
        paragraph = targetParagraphs[i].split(' ')
        for j in range(0, len(paragraph)):
            sum += len(paragraph[j])
            total += 1
    return sum / total, total


def basic_feature_extraction(inputs):
    new_inputs = []

    for data in inputs:
        hasMedia = 0 if len(data['postMedia']) == 0 else 1
        targetParagraphs = data['targetParagraphs']
        paragraph_length = average_paragraph_length(targetParagraphs)
        targetTitle = data['targetTitle']
        word_length, num_words = average_word_length(targetParagraphs)
        new_inputs.append([hasMedia, len(targetParagraphs), paragraph_length,
                           len(targetTitle), word_length, num_words])

    return new_inputs


def tfidf_features(path, ids):
    def load_file(filepath):
        with open(filepath) as f:
            return json.loads(f.read())
    try:
        data = load_file(os.path.join(path, 'tfidf_features_text.json'))
    except IOError:
        tfidf_feature_extraction(path)
        data = load_file(os.path.join(path, 'tfidf_features_text.json'))
    new_inputs = []
    for id in ids:
        new_inputs.append(data[str(id)])
    print "VOCAB SIZE:", len(new_inputs[0])
    return new_inputs


def tfidf_feature_extraction(path):
    with open(os.path.join(path, 'frequencies_text.json'), 'r') as f:
        entry = json.load(f)
    ids = entry['ids']
    tf = entry['term freqs']
    df = entry['doc freqs']
    new_inputs = {}
    for id in ids:
        tfidf_vocab = []
        counts = tf[id]
        for w in df:
            count = counts.get(w, 0)
            idf = -math.log(float(df[w]) / len(ids))
            tfidf_vocab.append(count*idf)
        new_inputs[str(id)] = tfidf_vocab
    count = 0
    with open(os.path.join(path, 'tfidf_features_text.json'), 'w') as f:
        f.write(json.dumps(new_inputs))

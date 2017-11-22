import json
import os
import math
import torchwordemb
import numpy as np
from string import punctuation


def get_word_ids(inputs, vocab, num_words=None):
    def get_tokens(inp):
        tokens = inp.lower().split(' ')
        special_chars = set(punctuation)
        special_chars.add("'s'")
        final_tokens = []
        for tok in tokens:
            has_special = False
            for c in special_chars:
                if c in tok:
                    has_special = True
                    toks = tok.split(c)
                    for t in toks:
                        final_tokens.append(t)
                        final_tokens.append(c)
                    del final_tokens[-1]
            if not has_special:
                final_tokens.append(tok)
        return final_tokens

    new_inputs = []
    for inp in inputs:
        inp = ''.join(inp['targetParagraphs'])
        # we reserve index 0 for padding, and 1 for unk
        # TODO actually figure out what to do with unknown keys
        new_input = [vocab.get(word, 1) + 2 for word in get_tokens(inp)]
        for tok in get_tokens(inp):
            if tok not in vocab:
                print tok
        new_input = [x for x in new_input if x is not 0]
        if num_words:
            if len(new_input) > num_words:
                new_input = new_input[:num_words]
            elif len(new_input) < num_words:
                new_input += [0] * (num_words - len(new_input))
        new_inputs.append(np.array(new_input, dtype=np.int32))
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

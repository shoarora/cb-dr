import json
import os
import math
import torchwordemb
import numpy as np
from string import punctuation

import re
from unidecode import unidecode as uni
from nltk import word_tokenize


def get_word_ids(inputs, vocab, num_words=None, target='text'):
    targets = {
        'text': 'targetParagraphs',
        'post': 'postText',
        'title': 'targetTitle'
    }
    target_key = targets[target]

    def get_tokens(inp):
        tokens = []
        inp_target = inp[target_key]
        if target == 'title':
            inp_target = [inp_target]
        for sent in inp_target:
            # convert all non-ascii to nearest ascii
            sent = uni(sent).lower()

            for token in word_tokenize(sent):
                # if token is the beginning or end of a quotation
                # drop the quotation. however, 's is OK.
                if token != "'s":
                    if len(token) > 1 and token[0] == "'":
                        token = token[1:]
                    if len(token) > 1 and token[-1] == "'":
                        token = token[:-1]
                tokens.append(token)
        return tokens

    new_inputs = []
    for inp in inputs:
        # need to increase default glove indices by 4 to make room
        # for special tokens [pad, unk, start, end]
        # we want the default for when a word isn't found to be 1
        # setting the default to -3 is gross, but this keeps runtime down
        new_input = [vocab.get(word, -3) + 4 for word in get_tokens(inp)]

        # add index for start token
        new_input = [2] + [x for x in new_input]

        # adjust to be num_words and add end token
        if num_words:
            if len(new_input) >= num_words:
                new_input = new_input[:num_words - 1] + [3]
            elif len(new_input) < num_words:
                new_input += [3] + [0] * (num_words - len(new_input) - 1)
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


#@Sho check this later. Choice represents the choice of what field you what to use (text, post, title)
#freq_floor represented the required document frequency for a term to be included in the vocab
def tfidf_features(path, ids, choice = 'text', freq_floor = 0):
    def load_file(filepath):
        with open(filepath) as f:
            return json.loads(f.read())
    save_title = 'tfidf_features_' + choice + '_' + str(freq_floor) + '_features.json'
    try:
        data = load_file(os.path.join(path, save_title ))
    except IOError:
        tfidf_feature_extraction(path,  save_title, choice, freq_floor)
        data = load_file(os.path.join(path, save_title ))
    new_inputs = []
    for id in ids:
        new_inputs.append(data[str(id)])
    print "VOCAB SIZE:", len(new_inputs[0])
    return new_inputs


def tfidf_feature_extraction(path,  save_title, choice, freq_floor = 0):
    with open(os.path.join(path, 'frequencies_' + choice + '.json'), 'r') as f:
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
            if df[w] < freq_floor:
                continue
            idf = -math.log(float(df[w]) / len(ids))
            tfidf_vocab.append(count*idf)
        new_inputs[str(id)] = tfidf_vocab
    count = 0
    print len(new_inputs[ids[0]])
    with open(os.path.join(path, save_title), 'w') as f:
        f.write(json.dumps(new_inputs))

def main():
    with open('../data/cb-small/instances.jsonl', 'r') as f:
         data = []
         for i in range(300):
             obj = json.loads(f.readline())
             entry = {'targetParagraphs':obj['targetParagraphs']}
             data.append(entry)

    with open('word_vec_test.json', 'w') as f:
         for entry in data:
             f.write(json.dumps(entry) + '\n')
    vocab, emb = torchwordemb.load_glove_text('glove.6B.50d.txt')
    #vocab = None
    #emb = None

    with open('word_vec_test.json', 'r') as f:
        inputs = [json.loads(line) for line in f]

    get_word_ids(inputs, vocab, emb)



if __name__ == '__main__':
    main()

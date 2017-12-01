import json
import os
import math
import torchwordemb
import numpy as np
from string import punctuation

import re
from unidecode import unidecode as uni
from nltk import word_tokenize

def get_word_ids(inputs, vocab, num_words=None):
    def test(inp):
        # convert all non-ascii to nearest ascii
        tokens = []
        for sent in inp['targetParagraphs']:
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

    def get_tokens(inp):
        inp = ''.join(inp['targetParagraphs'])
        #inp = inp['targetParagraphs'][0]
        tokens = inp.lower().split(' ')
        special_chars = set(punctuation)
        special_chars.add("'s")
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

    # #new_inputs = []
    # count = 0

    def hit_test(inputs, vocab, func):
        misses = []
        hits = []
        for inp in inputs:
            # we reserve 4 indices for pad, unk, start, and end
            # TODO actually figure out what to do with unknown keys
            for word in func(inp):
                if word in vocab:
                    hits.append(word)
                else:
                    misses.append(word)
        print 'hits: %d' % len(hits)
        print 'misses: %d' % len(misses)
        return (hits, misses)

    hits, misses = hit_test(inputs, vocab, get_tokens)
    t_hits, t_misses = hit_test(inputs, vocab, test)

    print filter(lambda x: x not in t_misses, misses)
    print
    
    print t_misses
    

    #     new_input = [vocab.get(word, 1) + 4 for word in get_tokens(inp)]
    #     for tok in get_tokens(inp):
    #         if tok not in vocab:
    #             print tok
    #     new_input = [3] + [x for x in new_input if x is not 0]
    #     if num_words:
    #         if len(new_input) > num_words:
    #             new_input = new_input[:num_words] + [4]
    #         elif len(new_input) < num_words:
    #             new_input += [4] + [0] * (num_words - len(new_input))
    #         else:
    #             new_input += [4]
    #     new_inputs.append(np.array(new_input, dtype=np.int32))
    # return new_inputs


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

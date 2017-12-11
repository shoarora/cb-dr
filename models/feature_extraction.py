from __future__ import division

import json
import os
import math
import torchwordemb
import numpy as np
from string import punctuation

import re
from unidecode import unidecode as uni
from nltk import word_tokenize

import spacy
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Tagger

from tqdm import tqdm

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

# function name is slightly misleading. technically, it checks if the
# first word is a number OR if first NAMED ENTITY is a number
# I think this method is actually more robust than just checking
# if the first word is a number.
def is_first_word_number(parsed):
    if parsed[0].pos_ == 'NUM':
        return True
    if len(parsed.ents) > 0 and \
            parsed.ents[0].label_ in ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
        return True
    return False

def match(parsed, window, condition):
    count = 0
    for i in range(len(parsed)):
        tokens = parsed[i:i+window]
        if len(tokens) < window:
            continue
        if condition(tokens):
            count += 1
    return count

def match_this_NN(parsed):
    def this_NN(tokens):
        return tokens[0].text.lower() in ['this', 'these'] and \
               tokens[1].tag_ == 'NN'
    return match(parsed, window=2, condition=this_NN)

def match_NNP_period(parsed):
    def NNP_period(tokens):
        return tokens[0].tag_ == 'NNP' and \
               tokens[1].text == '.'
    return match(parsed, window=2, condition=NNP_period)

def match_NUM_NP_VB(parsed):
    def NUM_NP_VB(tokens):
        return tokens[0].pos_ == 'NUM' and \
               tokens[1].pos_ == 'NOUN' and \
               tokens[2].pos_ == 'VERB'
    return match(parsed, window=3, condition=NUM_NP_VB)

# usage: match_tags(parsed_post, ['NNP', 'VBZ']) --> # of NNP VBZ matches
def match_tags(parsed, tags):
    def cond(tokens):
        return all(token.tag_ == tag for token, tag in zip(tokens, tags))
    return match(parsed, window=len(tags), condition=cond)

def top_60_feature_extraction(inputs):
    nlp = spacy.load('en')
    desired_labels = ['PERSON', 'NORP', 'ORG',
                      'GPE', 'LOC', 'PRODUCT', 'EVENT',
                      'WORK_OF_ART', 'LAW', 'LANGUAGE']
    tagger = Tagger(nlp.vocab, model=True)

    features = []
    for inp in tqdm(inputs):
        postStr = ' '.join(inp['postText'])

        parsed_post = nlp(postStr)
        parsed_title = nlp(inp['targetTitle'])
        keywords = [kw.strip().lower() for kw in inp['targetKeywords'].split(',')]

        # tokenize (also by punctuation)
        tokens_by_punc = word_tokenize(postStr)

        # get parts of speech
        TAG = [token.tag_ for token in parsed_post]

        # get word lengths in post
        lens = [len(token.text) for token in parsed_post]

        features.append([
            # 1 number of proper nouns
            match_tags(parsed_post, ['NNP']),
            # 2 readability of target paragraphs

            # 3 number of tokens
            len(tokens_by_punc),
            # 4 word length of post text
            len(parsed_post),
            # 5 POS 2-gram NNP NNP
            match_tags(parsed_post, ['NNP', 'NNP']),
            # 6 Whether the post starts with number
            1 if is_first_word_number(parsed_post) else 0,
            # 7 Average length of words in post
            np.mean(lens),
            # 8 Number of Prepositions / Subordinating Conjunction
            match_tags(parsed_post, ['IN']),
            # 9 POS 2-gram NNP 3rd person singular present Verb
            match_tags(parsed_post, ['NNP', 'VBZ']),
            # 10 POS 2-gram IN NNP
            match_tags(parsed_post, ['IN', 'NNP']),
            # 11 length of the longest word in post text
            max(lens),
            # 12 number of wh-adverb
            match_tags(parsed_post, ['WRB']),
            # 13 count POS pattern WRB

            # 14 number of single/mass nouns
            match_tags(parsed_post, ['NN']),
            # 15 count POS pattern NN

            # 16 whether the post text starts with 5W1H
            1 if parsed_post[0].tag_ in ['WDT', 'WP', 'WP$', 'WRB'] else 0,
            # 17 Whether exist Question Mark
            1 if '?' in postStr else 0,
            # 18 similarity between post and target title
            parsed_post.similarity(parsed_title),
            # 19 Count POS pattern this/these NN
            match_this_NN(parsed_post),
            # 20 Count POS pattern PRP

            # 21 Number of PRP
            match_tags(parsed_post, ['PRP']),
            # 22 Number of VBZ
            match_tags(parsed_post, ['VBZ']),
            # 23 POS 3-gram NNP NNP VBZ
            match_tags(parsed_post, ['NNP', 'NNP', 'VBZ']),
            # 24 POS 2-gram NN IN
            match_tags(parsed_post, ['NN', 'IN']),
            # 25 POS 3-gram NN IN NNP
            match_tags(parsed_post, ['NN', 'IN', 'NNP']),
            # 26 ratio of stop words in posttext
            len(filter(lambda x: x.is_stop, parsed_post)) / len(parsed_post),
            # 27 POS 2-gram NNP
            match_NNP_period(parsed_post),
            # 28 POS 2-gram PRP VBP
            match_tags(parsed_post, ['PRP', 'VBP']),
            # 29 Count POS pattern WP

            # 30 Number of WP
            match_tags(parsed_post, ['WP']),
            # 31 Count POS pattern DT

            # 32 Number of DT
            match_tags(parsed_post, ['DT']),
            # 33 POS 2-gram NNP IN
            match_tags(parsed_post, ['NNP', 'IN']),
            # 34 POS 3-gram IN NNP NNP
            match_tags(parsed_post, ['IN', 'NNP', 'NNP']),
            # 35 Number of POS
            match_tags(parsed_post, ['POS']),
            # 36 POS 2-gram IN IN
            match_tags(parsed_post, ['IN', 'IN']),
            # 37 Match between keywords and post
            len(filter(lambda x: x in postStr.lower(), keywords)),
            # 38 Number of ','
            len(filter(lambda x: x == ',', postStr)),
            # 39 POS 2-gram NNP NNS
            match_tags(parsed_post, ['NNP', 'NNS']),
            # 40 POS 2-gram IN JJ
            match_tags(parsed_post, ['IN', 'JJ']),
            # 41 POS 2-gram NNP POS
            match_tags(parsed_post, ['NNP', 'POS']),
            # 42 WDT
            match_tags(parsed_post, ['WDT']),
            # 43 Count POS pattern WDT

            # 44 POS 2-gram NN NN
            match_tags(parsed_post, ['NN', 'NN']),
            # 45 POS 2-gram NN NNP
            match_tags(parsed_post, ['NN', 'NNP']),
            # 46 POS 2-gram NNP VBD
            match_tags(parsed_post, ['NN', 'VBD']),
            # 47 Similarity between post and target paragraphs

            # 48 POS pattern RB
            match_tags(parsed_post, ['RB']),
            # 49 Number of RB

            # 50 POS 3-gram NNP NNP NNP
            match_tags(parsed_post, ['NNP', 'NNP', 'NNP']),
            # 51 POS 3-gram NNP NNP NN
            match_tags(parsed_post, ['NNP', 'NNP', 'NN']),
            # 52 Readability of target paragraphs

            # 53 Number of RBS
            match_tags(parsed_post, ['RBS']),
            # 54 Number of VBN
            match_tags(parsed_post, ['VBN']),
            # 55 POS 2-gram VBN IN
            match_tags(parsed_post, ['VBN', 'IN']),
            # 56 whether exist NUMBER NP VB
            match_NUM_NP_VB(parsed_post),
            # 57 POS 2-gram JJ NNP
            match_tags(parsed_post, ['JJ', 'NNP']),
            # 58 POS 3-gram NNP NN NN
            match_tags(parsed_post, ['NNP', 'NN', 'NN']),
            # 59 POS 2-gram DT NN
            match_tags(parsed_post, ['DT', 'NN']),
            # 60 whether exist EX
            1 if match_tags(parsed_post, ['EX']) > 1 else 0
        ])
    return features


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

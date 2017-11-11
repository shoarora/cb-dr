import json
import os
import math

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




def tfidf_feature_extraction(path):
    with open(os.path.join(path, 'frequencies.json'), 'r') as f:
        entry = json.load(f)
    ids = entry['ids']
    tf = entry['term freqs']
    df = entry['doc freqs']
    new_inputs = []
    for id in ids:
        tfidf_vocab = []
        counts = tf[id]
        for w in df:
            count = counts.get(w, 0) 
            idf = -math.log(float(df[w]) / len(ids))
            tfidf_vocab.append(count*idf)
        new_inputs.append(tfidf_vocab)
    with open(os.path.join(path, 'tfidf_features.json'), 'w') as f:
        f.write(json.dumps({'new_inputs': new_inputs}))


tfidf_feature_extraction('../../../clickbait17-train-170331')
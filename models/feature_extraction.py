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

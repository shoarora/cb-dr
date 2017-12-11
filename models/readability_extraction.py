from textstat.textstat import textstat


def flesch_score(text):
    try:
        if text == "":
            return 0
        return textstat.flesch_reading_ease(text)
    except:
        return 0
        
def article_flesch(targetParagraphs):
    sum = 0.0
    total = 0
    if len(targetParagraphs) == 0:
        return 0
    for i in range(0, len(targetParagraphs)):
        paragraph = targetParagraphs[i]
        sum += flesch_score(paragraph)
        total += 1
    if total == 0:
        return 0
    return sum / total

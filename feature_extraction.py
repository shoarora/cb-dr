import json
import numpy as np

def feature_extraction():

	data = []
	labels =[]

	def average_paragraph_length(targetParagraphs):
		sum = 0.0
		if len(targetParagraphs) == 0:
			return 0
		for i in range(0, len(targetParagraphs)):
			sum += len(targetParagraphs[i])
		return sum/len(targetParagraphs)

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
		return sum/total, total

	with open('clickbait17-train-170331/instances.jsonl') as json_data:
		for line in json_data:
			data.append(json.loads(line))
	with open('clickbait17-train-170331/truth.jsonl') as json_data:
		for line in json_data:
			labels.append(json.loads(line))
	data_features = {}
	features = {'hasMedia' : 0, 'numParagraphs' : 1, 'avgParagraphLength' : 2, 'titleLen': 3, 'wordLen': 4, 'num_words': 5}
	ids = []
	for i in range(0, len(data)):
		hasMedia = 0 if len(data[i]['postMedia']) == 0 else 1
		targetParagraphs = data[i]['targetParagraphs']
		paragraph_length = average_paragraph_length(targetParagraphs)
		targetTitle = data[i]['targetTitle']
		id =  int(data[i]['id'])
		word_length, num_words=  average_word_length(targetParagraphs)
		data_features[id] = [hasMedia, len(targetParagraphs), paragraph_length, len(targetTitle), word_length, num_words]
		ids.append(id)
	id_and_labels = {}
	for i in range(0, len(labels)):
		id =  int(labels[i]['id'])
		binary_label = 0 if labels[i]['truthClass'] == 'no-clickbait' else 1
		id_and_labels[id] = [labels[i]['truthMedian'], binary_label]

	num_samples = len(ids)
	num_features = len(data_features[ids[0]])

	X = np.zeros((num_samples, num_features))
	Y = np.zeros((num_samples, 1))
	for i in range(0, len(ids)):
		id = ids[i]
		X[i] = np.array(data_features[id])
		Y[i] = np.array(id_and_labels[id][1])
	return X, Y
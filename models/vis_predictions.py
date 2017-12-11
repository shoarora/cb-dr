def load_json(path):
    data = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            id = entry['id']
            data[id] = entry
    return data

def plot_on_line(pred, truth):
	for id in pred:
		if random.random() < .75: #to plot fewer points
			continue
		if truth[id]['truthClass'] == 'no-clickbait':
			plt.plot( [id]['clickbaitScore'], 0, 'bx')
		else:
			plt.plot([id]['clickbaitScore'], 0, 'ro')

	plt.show()
	
def plot_against(pred, truth):
	for id in train_pred:
		if truth[id]['truthClass'] == 'no-clickbait':
			plt.plot(truth[id]['truthMedian'], pred[id]['clickbaitScore'], 'bx')
		else:
			plt.plot(truth[id]['truthMedian'], pred[id]['clickbaitScore'], 'ro')

	plt.show()

def cluster_analysis(pred, truth):
	clusters = {}
	for id in pred:
		if not truth[id]['truthMedian']	in clusters:
			clusters[truth[id]['truthMedian']] = []
		clusters[truth[id]['truthMedian']].append(pred[id]['clickbaitScore'])
	for cluster in clusters:
		sum = 0.0
		for i in clusters[cluster]:
			sum += i
		print cluster
		print sum  / len(clusters[cluster])
 
path = 'clickbait17-validation-170630/'
truth = load_json(path+'truth.jsonl')
train_path = 'train32_predictions.json'
train_pred = load_json(train_path)
dev_path = 'dev32_predictions.json'
dev_pred = load_json(dev_path)

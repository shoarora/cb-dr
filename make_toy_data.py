import json

SIZE = 100

with open('data/cb-small/truth.jsonl', 'r') as f:
    truths = [json.loads(line) for line in f]

with open('data/cb-small/instances.jsonl', 'r') as f:
    instances = [json.loads(line) for line in f]


pos = []
neg = []
for i in range(len(truths)):
    if truths[i]['truthClass'] == u'no-clickbait':
        neg.append(i)
    else:
        pos.append(i)

with open('data/cb-mini/instances.jsonl', 'w') as f:
    for i in range(SIZE):
        f.write(json.dumps(instances[neg[i]]) + '\n')
        f.write(json.dumps(instances[pos[i]]) + '\n')

with open('data/cb-mini/truth.jsonl', 'w') as f:
    for i in range(SIZE):
        f.write(json.dumps(truths[neg[i]]) + '\n')
        f.write(json.dumps(truths[pos[i]]) + '\n')



import json

DIR_FILE = 'data/cb-big/'


with open(DIR_FILE + 'instances.jsonl', 'r') as f:
    instances = [json.loads(line) for line in f]

with open(DIR_FILE + 'truth.jsonl', 'r') as f:
    truths = {}
    for line in f:
        obj = json.loads(line)
        truths[obj['id']] = obj

#with open(DIR_FILE + 'truth_order.jsonl', 'w') as f:
new_truths = []
for ins in instances:
    new_truths.append(truths[ins['id']])

with open(DIR_FILE + 'ordered_truth.jsonl', 'w') as f:
    for truth in new_truths:
        f.write(json.dumps(truth) + '\n')

with open(DIR_FILE + 'ordered_truth.jsonl', 'r') as f:
    truths = [json.loads(line) for line in f]
    for t, ins in zip(truths, instances):
        assert t['id'] == ins['id']

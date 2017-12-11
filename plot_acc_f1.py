import json
import os
import math
import matplotlib.pyplot as plt
import random as random 

results_path = 'clickbait17-validation-170630/results3l1/'

acc = []
f1 = []
next_f1 = False
next_acc = False

for i in range(0, 32):

	filename = 'dev' + str(i) + '_output.prototext'
	
	with open(path+filename) as f:
		for line in f:
			if next_f1:
				new = line.split("\"")
				f1.append(float(new[1]))
				next_f1 = False
			if next_acc:
				new = line.split("\"")
				acc.append(float(new[1]))
				next_acc = False
			if 'F1' in line:
				next_f1 = True
			if 'Acc' in line:
				next_acc = True
plt.plot(acc)
plt.show()
plt.plot(f1)
plt.show()

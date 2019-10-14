from sklearn.model_selection import train_test_split
import os
import collections
cwd = os.getcwd()

with open(cwd+'/firsttrain.txt', 'r') as f, open(cwd+'/firsttest.txt', 'r') as other:
	firsttrain =  f.readlines()
	firsttest = other.readlines()
	thislist = firsttrain+firsttest
	duplicates = [item+" "+str(count) for item, count in collections.Counter(thislist).items() if count > 1]
with open(cwd+'/challenge-data/train.txt', 'r',errors = 'ignore') as f, open(cwd+'/training.txt', 'w') as train, open(cwd+'/valid.txt', 'w') as valid:
	
	X = []
	for item in f.readlines():
		X.extend(item.split("\t"))
	firstlines = X[::2]
	corrupt = X[1::2]
	beento = []
	for sent in firsttrain:
		if sent in duplicates:
			if sent in beento:
				train.write(sent + "\t"+corrupt[firstlines.index(sent[:-1])])
			else:
				train.write(sent + "\t"+corrupt[firstlines.index(sent[:-1],1)])
		else: 
			train.write(sent + "\t"+corrupt[firstlines.index(sent[:-1])])
	for sent in firsttest:
		if sent in duplicates:
			if sent in beento:
				valid.write(sent + "\t"+corrupt[firstlines.index(sent[:-1])])
			else:
				valid.write(sent + "\t"+corrupt[firstlines.index(sent[:-1],1)])
		else: 
			valid.write(sent + "\t"+corrupt[firstlines.index(sent[:-1])])


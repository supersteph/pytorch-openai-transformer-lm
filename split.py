from sklearn.model_selection import train_test_split
import os
cwd = os.getcwd()
text_file = open(cwd+"/challenge-data/train.txt", "r", encoding='utf-8',errors = 'ignore')
xTrain, xTest = train_test_split(text_file.readlines(), test_size = 0.2)

with open(cwd+'/firsttrain.txt', 'w') as f:
	for item in xTrain:
		a = item.split("\t")[0]
		f.write("%s\n" % a)
with open(cwd+'/firsttest.txt', 'w') as f:
	for item in xTest:
		a = item.split("\t")[0]
		f.write("%s\n" % a)
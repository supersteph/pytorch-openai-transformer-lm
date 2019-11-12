import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _getData(path):
    with open(path, encoding='utf_8', errors = 'ignore') as f:
        X = []
        for item in f.readlines():
            if item.count('\t')>1:
                if item[0] == "\t":
                    X.append("x")
                    X.append("")
                else:
                    X.append("")
                    X.append("x")
            elif item[0] == " ":
                X.append("x")
                X.append("")
            elif "\t " in item:
                X.append("")
                X.append("x")
            else:
                X.extend(item.split("\t"))
        return [sent for sent in X[::2]], [sent for sent in X[1::2]]

def getData(data_dir, n_train=1497, n_valid=374):
    first, second = _getData(os.path.join(os.getcwd(), 'challenge-data/train.txt'))
    #teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    trX1 = []
    trX2 = []
    for i in range(len(first)):
        trX1.append(first[i].strip())
        trX2.append(second[i].strip())


    
    return trX1, trX2

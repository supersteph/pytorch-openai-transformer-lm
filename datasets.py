import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _getData(path):
    with open(path, encoding='utf_8', errors = 'ignore') as f:
        return f.readlines()

def getData(data_dir, n_train=1497, n_valid=374):
    trStuff = _getData(os.path.join(data_dir, 'halftrain.txt'))
    vaStuff = _getData(os.path.join(data_dir, 'firsttest.txt'))
    #teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    trX = []
    for s in trStuff:
        trX.append(s.strip())

    vaX = []
    for s in vaStuff:
        vaX.append(s.strip())

    
    return trX, vaX

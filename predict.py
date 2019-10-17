#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from datasets import getData
from model_pytorch import LMModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import LMLossCompute

def transform_roc(X):
    length = len(X)
    xmb = np.zeros((1, length+2, 2), dtype=np.int32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']

    x = [start] + X + [delimiter]
    xmb[0, :, 0] = x

    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + length+2)
    return xmb


def iter_predict(x):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        XMB = torch.tensor(x, dtype=torch.long).to(device)
        lm_logits = dh_model(XMB)
        logits=lm_logits.to("cpu").numpy()
    return logits

def letter(first, second):
    for i, word in enumerate(first):
        if set(word) == set(second[i]) and word != second[i]:
            return True
    return False

def calculate(bpe, logit):
    sum = 0
    for i in range(len(bpe)-1):
        sum += logit[i, bpe[i+1]]
    return sum

def predict(firstsents, secondsents, firstbpes, secondbpes):
    arr = []
    for i in range(len(firstsents)):
        firstsent = firstsents[i]
        secondsent = secondsents[i]
        firstbpe = firstbpes[i]
        secondbpe = secondbpes[i]
        firstX = transform_roc(firstbpe)
        secondX = transform_roc(secondbpe)
        #do the weird space thing
        if len(firstsent) != len(secondsent):
            arr.append(len(firstsent) > len(secondsent))
        elif letter(firstsent, secondsent):
            arr.append(len(firstbpe) > len(secondbpe))
        else:
            logits1 = iter_predict(firstX)
            logits2 = iter_predict(secondX)
            arr.append(calculate(firstbpe, logits1)>calculate(secondbpe, logits2))
    return arr;


argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=64)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    print("Encoding dataset...")
    firstsent, secondsent = getData(data_dir, n_valid=args.n_valid)
    firstbpe, secondbpe = encode_dataset(*(firstsent, secondsent), encoder = text_encoder)
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)

    n_special = 3
    max_len = n_ctx // 2 - 2
    n_ctx = 95;
    vocab = n_vocab + n_special + n_ctx
    print(max(firstbpe, key=len))
    print(len(max(secondbpe, key=len)))
    n_train = len(firstsent)
    n_valid = len(secondsent)

    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    dh_model = LMModel(args, vocab, n_ctx)

    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)
    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)
    n_updates = 0
    n_epochs = 0

    path = os.path.join(save_dir, desc, 'best_params')
    dh_model.load_state_dict(torch.load(path))
    arr = predict(firstsent, secondsent, firstbpe, secondbpe)
    with open(os.path.join(os.getcwd(), 'part1.txt'), 'w') as w:
        for pred in arr:
            if pred:
                w.write('A\n')
            else:
                w.write('B\n')

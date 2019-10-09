#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time
#import psutil

#from analysis import rocstories as rocstories_analysis
from datasets import getData
from model_pytorch import LMModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import LMLossCompute

from pympler import muppy, summary

def transform_roc(X):
    n_batch = len(X)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, x, in enumerate(X):
        x = [start] + x[:max_len] + [delimiter]
        l = len(x)
        xmb[i, 0:l, 0] = x
        mmb[i, 0:l] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    print(Xs.shape)
    with torch.no_grad():
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            print(XMB.size())
            MMB = torch.tensor(mmb).to(device)
            lm_logits = dh_model(XMB)
            lm_logits *= n
            lm_losses = compute_loss_fct(XMB, MMB, lm_logits, only_return_losses=True)
            lm_losses *= n
            print(lm_logits.size())
            #lm_logits is (4096 40738)
            logits.append(lm_logits.data.cpu().numpy())
            #print(len(logits))
            cost += lm_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost



def log(save_dir, desc):
    global best_score
    print("Logging")
    #trX is 2000 257 2
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid])
    print("valid")
    va_logits, va_cost = iter_apply(vaX, vaM)
    tr_cost = tr_cost / len(trX[:n_valid])
    va_cost = va_cost / n_valid
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost)
    print('%d %d %.3f %.3f' % (n_epochs, n_updates, tr_cost, va_cost))
    if submit:
        score = va_cost
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def run_epoch():
    for xmb, mmb in iter_data(*shuffle(trX, trM, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits = dh_model(XMB)
        compute_loss_fct(XMB, MMB, lm_logits)
        n_updates += 1
        if n_updates in [2, 100, 1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)


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
    start = time.time()

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
    start_time = time.time()
    print("Encoding dataset...")
    (trX, vaX) = encode_dataset(*getData(data_dir, n_valid=args.n_valid),
                                        encoder=text_encoder)
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    n_special = 3
    max_len = n_ctx // 2 - 2
    n_ctx = min(max(
        [len(x[:max_len])  for x in trX]
        + [len(x[:max_len]) for x in vaX]
        ) + 3, n_ctx)
    vocab = n_vocab + n_special + n_ctx
    trX, trM = transform_roc(trX)
    vaX, vaM = transform_roc(vaX)

    n_train = len(trX)
    n_valid = len(vaX)

    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    dh_model = LMModel(args, vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = LMLossCompute(criterion, model_opt)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)
    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)
    n_updates = 0
    n_epochs = 0
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0

    print(time.time()-start_time)
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        log(save_dir, desc)
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
    #     dh_model.load_state_dict(torch.load(path))
    #     predict(dataset, args.submission_dir)
    #     if args.analysis:
    #         rocstories_analysis(data_dir, os.path.join(args.submission_dir, 'ROCStories.tsv'),
    #                             os.path.join(log_dir, 'rocstories.jsonl'))
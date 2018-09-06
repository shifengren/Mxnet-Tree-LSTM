#encoding=utf-8

import os
import logging
import numpy as np
import random
from tqdm import tqdm
import argparse, pickle, math

import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn
from mxnet.gluon.parameter import Parameter
from mxnet import autograd as ag

from child_sum_tree_lstm import *

logging.basicConfig(level=logging.INFO)

# Training setting
use_gpu = False
optimizer = "AdaGrad"
seed = 123
batch_size = 25
training_batches_per_epoch = 10
learning_rate = 0.01
weight_decay = 0.0001
epoches = 1
rnn_hidden_size, sim_hidden_size, num_classes = 150, 100, 5

# initialization
context = [mx.gpu(0) if use_gpu else mx.cpu()]

#seeding 
mx.random.seed(seed)
np.random.seed(seed)
random.seed(seed)

# Read Dataset
def verified(file_path, sha1hash):
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    matched = sha1.hexdigest() == sha1hash
    if not matched:
        logging.warn('Found hash mismatch in file {}, possibly due to incomplete download.'
                     .format(file_path))
    return matched

data_file_name = 'data/tree_lstm_dataset-3d85a6c4.cPickle'
data_file_hash = '3d85a6c44a335a33edc060028f91395ab0dcf601'
if not os.path.exists(data_file_name) or not verified(data_file_name, data_file_hash):
    from mxnet.test_utils import download
    download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/%s'%data_file_name,
             overwrite=True)

with open(data_file_name, 'rb') as f:
    train_iter, dev_iter, test_iter, vocab = pickle.load(f)

logging.info('==> SICK vocabulary size : %d ' % vocab.size)
logging.info('==> Size of train data   : %d ' % len(train_iter))
logging.info('==> Size of dev data     : %d ' % len(dev_iter))
logging.info('==> Size of test data    : %d ' % len(test_iter))

# get network
net = SimilarityTreeLSTM(sim_hidden_size, rnn_hidden_size, vocab.size, vocab.embed.shape[1], num_classes)
# use pearson correlation and mean-square error for evaluation
metric = mx.metric.create(['pearsonr', 'mse'])

# the prediction from the network is log-probability vector of each score class
# so use the following function to convert scalar score to the vector
# e.g 4.5 -> [0, 0, 0, 0.5, 0.5]
def to_target(x):
    target = np.zeros((1, num_classes))
    ceil = int(math.ceil(x))
    floor = int(math.floor(x))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - x
        target[0][ceil-1] = x - floor
    return nd.array(target)

# and use the following to convert log-probability vector to score
def to_score(x):
    levels = nd.arange(1, 6, ctx=x.context)
    res = nd.sum(levels * nd.exp(x), axis=1)
    return [nd.reshape(res, (-1,1))]

# when evaluating in validation mode, check and see if pearson-r is improved
# if so, checkpoint and run evaluation on test dataset
def test(ctx, data_iter, best, mode='validation', num_iter=-1):
    data_iter.reset()
    samples = len(data_iter)
    data_iter.set_context(ctx[0])
    preds = []
    labels = [mx.nd.array(data_iter.labels, ctx=ctx[0]).reshape((-1,1))]
    for _ in tqdm(range(samples), desc='Testing in {} mode'.format(mode)):
        l_tree, l_sent, r_tree, r_sent, label = data_iter.next()
        z = net(mx.nd, l_sent, r_sent, l_tree, r_tree)
        preds.append(z)

    preds = to_score(mx.nd.concat(*preds, dim=0))
    metric.update(preds, labels)
    names, values = metric.get()
    metric.reset()
    for name, acc in zip(names, values):
        logging.info(mode+' acc: %s=%f'%(name, acc))
        if name == 'pearsonr':
            test_r = acc
    if mode == 'validation' and num_iter >= 0:
        if test_r >= best:
            best = test_r
            logging.info('New optimum found: {}.'.format(best))
        return best

def train(epoch, train_data, dev_data, ctx):
    # initialization with context
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx[0])
    net.embed.weight.set_data(vocab.embed.as_in_context(ctx[0]))
    train_data.set_context(ctx[0])
    dev_data.set_context(ctx[0])

    # set up trainer for optimizing the network.
    trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': learning_rate, 'wd': weight_decay})

    best_r = -1
    Loss = gluon.loss.KLDivLoss()
    for i in range(epoch):
        train_data.reset()
        num_samples = min(len(train_data), training_batches_per_epoch*batch_size)
        # collect predictions and labels for evaluation metrics
        preds = []
        labels = [mx.nd.array(train_data.labels[:num_samples], ctx=ctx[0]).reshape((-1,1))]
        for j in tqdm(range(num_samples), desc='Training epoch {}'.format(i)):
            # get next batch
            l_tree, l_sent, r_tree, r_sent, label = train_data.next()
            # use autograd to record the forward calculation
            with ag.record():
                # forward calculation. the output is log probability
                z = net(mx.nd, l_sent, r_sent, l_tree, r_tree)
                # calculate loss
                loss = Loss(z, to_target(label).as_in_context(ctx[0]))
                # backward calculation for gradients.
                loss.backward()
                preds.append(z)
            # update weight after every batch_size samples
            if (j+1) % batch_size == 0:
                trainer.step(batch_size)

        # translate log-probability to scores, and evaluate
        preds = to_score(mx.nd.concat(*preds, dim=0))
        metric.update(preds, labels)
        names, values = metric.get()
        metric.reset()
        for name, acc in zip(names, values):
            logging.info('training acc at epoch %d: %s=%f'%(i, name, acc))
        best_r = test(ctx, dev_data, best_r, num_iter=i)

train(epoches, train_iter, dev_iter, context)


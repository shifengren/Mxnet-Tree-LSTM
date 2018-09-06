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

logging.basicConfig(level=logging.INFO)


class Tree(object):
    """
    Store tree-structure data
    """
    def __init__(self, idx):
        self.children = []
        self.idx = idx

    def __repr__(self):
        if self.children:
            return '{0}: {1}'.format(self.idx, str(self.children))
        else:
            return str(self.idx)


class ChildSumLSTMCell(nn.Block):
    def __init__(self, hidden_size, 
                i2h_weight_initializer=None,
                hs2h_weight_initializer=None,
                hc2h_weight_initializer=None,
                i2h_bias_initializer="zeros",
                hs2h_bias_initializer="zeros",
                hc2h_bias_initializer="zeros",
                input_size=0, prefix=None, params=None):
        super(ChildSumLSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            # 4: input gate, forget gate, output gate, transform
            self._i2h_weight = self.params.get("i2h_weight", 
                                                shape=(4*self._hidden_size, self._input_size),
                                                init=i2h_weight_initializer)
            # 3: input gate, output gate, transform (forget gate should be process seperately)
            self._hs2h_weight = self.params.get("hs2h_weight", 
                                                shape=(3*self._hidden_size, self._hidden_size), 
                                                init=hs2h_weight_initializer)
            # For forget gate
            self._hc2h_weight = self.params.get("hc2h_weight", 
                                                shape=(self._hidden_size, self._hidden_size),
                                                init=hc2h_weight_initializer)
            self._i2h_bias = self.params.get("i2h_bias",
                                            shape=(4*self._hidden_size,),
                                            init=i2h_bias_initializer)
            self._hs2h_bias = self.params.get("hs2h_bias",
                                            shape=(3*self._hidden_size,),
                                            init=hs2h_bias_initializer)
            self._hc2h_bias = self.params.get("hc2h_bias",
                                            shape=(self._hidden_size,),
                                            init=hc2h_bias_initializer)
    def forward(self, F, inputs, tree):
        children_outputs = [self.forward(F, inputs, child) for child in tree.children] # resursively, Top->Bottom
        if children_outputs:
            _, children_states = zip(*children_outputs)
        else:
            children_states = None
        with inputs.context as ctx: # Bottom->Top
            return self._forward(F, F.expand_dims(inputs[tree.idx], axis=0), # ? 
                                children_states,
                                self._i2h_weight.data(ctx),
                                self._hs2h_weight.data(ctx),
                                self._hc2h_weight.data(ctx),
                                self._i2h_bias.data(ctx),
                                self._hs2h_bias.data(ctx),
                                self._hc2h_bias.data(ctx));

    def _forward(self, F, inputs, children_states, i2h_weight, hs2h_weight, hc2h_weight, 
                i2h_bias, hs2h_bias, hc2h_bias):
        # N: number of observations
        # C: hidden state size
        # K: number of children
        
        # FC from inputs to hidden on input(i), forget(f), output(o) gate and transform operation(u)
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias, num_hidden=self._hidden_size * 4) # (N, 4*C)
        i2h_slice = F.split(i2h, num_outputs = 4) # (4, N, C)
        i2h_iuo = F.concat(*[i2h_slice[i] for i in [0, 2, 3]], dim=1) # (N, 3*C)

        if children_states:
            # Sum of children hidden states
            hs_sum = F.add_n( *[state[0] for state in children_states] ) # (N, C)
            # Concate children hidden states
            hs_concat = F.concat( *[F.expand_dims(state[0], axis=1) for state in children_states], dim=1) #(N, K, C)
            # Concate children cell memory states
            cs_concat = F.concat( *[F.expand_dims(state[1], axis=1) for state in children_states], dim=1) #(N, K, C)

            # Compute activation output on forget gate
            i2h_f = i2h_slice[1] # (N, C)
            f_act = i2h_f + hc2h_bias + F.dot(hs_concat, hc2h_weight) # (N, K, C)
            f_gate = F.Activation(f_act, act_type="sigmoid") # (N, K, C)
        else:
            # Summation of children hidden states are zeros on leaf node (no input hidden state)
            hs_sum = F.zeros_like(i2h_slice[0]) # (N, C)
        
        # FC, From sum of children states to hidden state on i, u, o gate
        hs2h_iuo = F.FullyConnected(data=hs_sum, weight=hs2h_weight, bias=hs2h_bias, num_hidden=3 * self._hidden_size) #(N, 3*C)
        i2h_iuo = i2h_iuo + hs2h_iuo
        iuo_act_slices = F.split(i2h_iuo, num_outputs=3) # (3, N, C)
        i_act, u_act, o_act = [act for act in iuo_act_slices] # (N,C)

        # Gate output
        i_gate = F.Activation(i_act, act_type="sigmoid")
        in_transform = F.Activation(u_act, act_type="tanh")
        o_gate = F.Activation(o_act, act_type="sigmoid") # (N, C)

        # Final Cell memory and hidden state
        next_c = i_gate * in_transform
        if children_states:
            next_c = F.sum(f_gate * cs_concat, axis=1) + next_c # (N, C)
        next_h = o_gate * F.Activation(next_c, act_type="tanh") # (N, C)
        return next_h, [next_h, next_c]


class Similarity(nn.Block):
    """
    ?
    """
    def __init__(self, sim_hidden_size, rnn_hidden_size, num_classes):
        super(Similarity, self).__init__()
        with self.name_scope():
            self._wh = nn.Dense(sim_hidden_size, in_units=2*rnn_hidden_size)
            self._wp = nn.Dense(num_classes, in_units=sim_hidden_size)
    
    def forward(self, F, lvec, rvec):
        # lvec, rvec will be tree-lstm cell state at root
        mult_dist = F.broadcast_mul(lvec, rvec)
        abs_dist = F.abs(F.add(lvec, -rvec))
        vec_dist = F.concat(*[mult_dist, abs_dist], dim=1)
        out = F.log_softmax(self._wp(F.sigmoid(self._wh(vec_dist))))
        return out

class SimilarityTreeLSTM(nn.Block):
    """
    Similarity tree lstm.
    """
    def __init__(self, sim_hidden_size, rnn_hidden_size, embed_in_size, embed_dim, num_classes):
        super(SimilarityTreeLSTM, self).__init__()
        with self.name_scope():
            self.embed = nn.Embedding(embed_in_size, embed_dim)
            self.child_sum_tree_lstm = ChildSumLSTMCell(rnn_hidden_size, input_size=embed_dim)
            self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)
    def forward(self, F, l_inputs, r_inputs, l_tree, r_tree):
        l_inputs = self.embed(l_inputs)
        r_inputs = self.embed(r_inputs)
        lstate = self.child_sum_tree_lstm(F, l_inputs, l_tree)[1][1] # utilize final cell memory
        rstate = self.child_sum_tree_lstm(F, r_inputs, r_tree)[1][1]
        output = self.similarity(F, lstate, rstate)
        return output


class Vocab(object):
    # constants for special tokens: padding, unknown, and beginning/end of sentence.
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD = '<blank>', '<unk>', '<s>', '</s>'
    def __init__(self, filepaths=[], embedpath=None, include_unseen=False, lower=False):
        self.idx2tok = []
        self.tok2idx = {}
        self.lower = lower
        self.include_unseen = include_unseen

        self.add(Vocab.PAD_WORD)
        self.add(Vocab.UNK_WORD)
        self.add(Vocab.BOS_WORD)
        self.add(Vocab.EOS_WORD)

        self.embed = None

        for filename in filepaths:
            logging.info('loading %s'%filename)
            with open(filename, 'r') as f:
                self.load_file(f)
        if embedpath is not None:
            logging.info('loading %s'%embedpath)
            with open(embedpath, 'r') as f:
                self.load_embedding(f, reset=set([Vocab.PAD_WORD, Vocab.UNK_WORD, Vocab.BOS_WORD,
                                                  Vocab.EOS_WORD]))

    @property
    def size(self):
        return len(self.idx2tok)

    def get_index(self, key):
        return self.tok2idx.get(key.lower() if self.lower else key,
                                Vocab.UNK)

    def get_token(self, idx):
        if idx < self.size:
            return self.idx2tok[idx]
        else:
            return Vocab.UNK_WORD

    def add(self, token):
        token = token.lower() if self.lower else token
        if token in self.tok2idx:
            idx = self.tok2idx[token]
        else:
            idx = len(self.idx2tok)
            self.idx2tok.append(token)
            self.tok2idx[token] = idx
        return idx

    def to_indices(self, tokens, add_bos=False, add_eos=False):
        vec = [BOS] if add_bos else []
        vec += [self.get_index(token) for token in tokens]
        if add_eos:
            vec.append(EOS)
        return vec

    def to_tokens(self, indices, stop):
        tokens = []
        for i in indices:
            tokens += [self.get_token(i)]
            if i == stop:
                break
        return tokens

    def load_file(self, f):
        for line in f:
            tokens = line.rstrip('\n').split()
            for token in tokens:
                self.add(token)

    def load_embedding(self, f, reset=[]):
        vectors = {}
        for line in tqdm(f.readlines(), desc='Loading embeddings'):
            tokens = line.rstrip('\n').split(' ')
            word = tokens[0].lower() if self.lower else tokens[0]
            if self.include_unseen:
                self.add(word)
            if word in self.tok2idx:
                vectors[word] = [float(x) for x in tokens[1:]]
        dim = len(vectors.values()[0])
        
        self.embed = mx.nd.array([vectors[tok] if tok in vectors and tok not in reset
                                  else [0.0]*dim for tok in self.idx2tok])



class SICKDataIter(object):
    def __init__(self, path, vocab, num_classes, shuffle=True):
        super(SICKDataIter, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.l_sentences = []
        self.r_sentences = []
        self.l_trees = []
        self.r_trees = []
        self.labels = []
        self.size = 0
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            mask = list(range(self.size))
            random.shuffle(mask)
            self.l_sentences = [self.l_sentences[i] for i in mask]
            self.r_sentences = [self.r_sentences[i] for i in mask]
            self.l_trees = [self.l_trees[i] for i in mask]
            self.r_trees = [self.r_trees[i] for i in mask]
            self.labels = [self.labels[i] for i in mask]
        self.index = 0

    def next(self):
        out = self[self.index]
        self.index += 1
        return out

    def set_context(self, context):
        self.l_sentences = [a.as_in_context(context) for a in self.l_sentences]
        self.r_sentences = [a.as_in_context(context) for a in self.r_sentences]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        l_tree = self.l_trees[index]
        r_tree = self.r_trees[index]
        l_sent = self.l_sentences[index]
        r_sent = self.r_sentences[index]
        label = self.labels[index]
        return (l_tree, l_sent, r_tree, r_sent, label)

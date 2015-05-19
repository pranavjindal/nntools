from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import os
import time
import sys


import lasagne


def load_data(file_name, vocab_map, vocab_idx):
    """
    Loads Penn Tree files from https://github.com/wojzaremba/lstm

    Parameters
    ----------
    file_name : str
        Path to datafile
    vocab_map : dictionary
        Dictionary mapping words to integers
    vocab_idx : one element list
        Current vocabolary index

    Returns
    -------
    Returns an array with the words specified in file_name encoded as integers.
    Note that the funtion has the sideeffects that vocab_map and vocab_idx are
    updated.

    Notes
    -----
    This is python port of the LUA function load_data in
    https://github.com/wojzaremba/lstm/blob/master/data.lua
    """

    def process_line(line):
        line = line.lstrip()
        line = line.replace('\n', '<eos>')
        words = line.split(" ")
        if words[-1] == "":
            del words[-1]
        return words

    words = []
    with open(file_name, 'rb') as f:
        for line in f.readlines():
            words += process_line(line)

    n_words = len(words)
    print("Loaded %i words from %s" % (n_words, file_name))

    x = np.zeros(n_words)
    for wrd_idx, wrd in enumerate(words):
        if wrd not in vocab_map:
            vocab_map[wrd] = vocab_idx[0]
            vocab_idx[0] += 1
        x[wrd_idx] = vocab_map[wrd]
    return x.astype('int32')


def traindata(model_seq_len, vocab_map, vocab_idx):
    x = load_data(os.path.join(folder, "ptb.train.txt"),
                  vocab_map, vocab_idx)
    n_batches = x.shape[0] // model_seq_len
    n_samples = n_batches * model_seq_len
    x = x[:n_samples]
    return x.reshape((n_batches, model_seq_len))


def load_data_masked(model_seq_len, batch_size, vocab_map, vocab_idx, fn):
    x = load_data(os.path.join(folder, fn),
                  vocab_map, vocab_idx)

    # we pad so x is i multiple of batch_size*model_seq_len.
    # the +1 ensures that we can shift when we need to extract targets.
    n_samples_out = np.ceil(x.shape[0] / float(model_seq_len * batch_size)) * \
                    model_seq_len * batch_size + 1

    pad_len = n_samples_out - x.shape[0]
    pad = np.zeros((pad_len), dtype=x.dtype)

    mask = np.zeros(n_samples_out, dtype=np.bool)
    mask[:x.shape[0]] = True
    x = np.concatenate([x, pad])
    return x, mask


def validdata(model_seq_len, vocab_map, vocab_idx):
    x = load_data(os.path.join(folder, "ptb.valid.txt"),
                  vocab_map, vocab_idx)
    n_batches = x.shape[0] // model_seq_len
    n_samples = n_batches * model_seq_len
    x = x[:n_samples]
    return x.reshape((n_batches, model_seq_len))


def testdata(model_seq_len, batch_size, vocab_map, vocab_idx):
    return load_data_masked(
        model_seq_len, batch_size, vocab_map, vocab_idx, "ptb.test.txt")


folder = 'penntree'
vocab_map = {}
vocab_idx = [0]

BATCH_SIZE = 20
MODEL_SEQ_LEN = 20
x_train = traindata(MODEL_SEQ_LEN, vocab_map, vocab_idx)
x_test, mask_test = testdata(MODEL_SEQ_LEN, BATCH_SIZE, vocab_map, vocab_idx)
x_valid = validdata(MODEL_SEQ_LEN, vocab_map, vocab_idx)

LARGE_MODEL = True
vocab_size = vocab_idx[0]
TOL = 1e-6
lr = 1e-3
eval_freq = 10000000   # number of batches between eval


if LARGE_MODEL:
    ini = lasagne.init.Uniform(0.04)
    rec_num_units = 1500
    embedding_size = rec_num_units
    num_epochs = 100
    dropout_frac = 0.65
else:
    ini = lasagne.init.Uniform(0.1)
    rec_num_units = 200
    embedding_size = rec_num_units
    num_epochs = 10
    dropout_frac = 0



print("-" * 80)
print("Vocab size:s", vocab_size)
print("Data shapes")
print("Train data:", x_train.shape)
print("Test data :", x_test.shape)
print("Valid data:", x_valid.shape)
print("-" * 80)

# BUILDING THE MODEL
# ------------------
# For a languge model we want to use the previous prediction as input in t+1.
# In Lasagne you do that by giving an output network in the output_network arg.
#
# Model structure:
#
#    embedding -> LSTM1 --> LSTM2 --> output network------> predictions


def build_model(batch_size, model_seq_len):
    l_inp = lasagne.layers.InputLayer((batch_size, model_seq_len))

    l_emb = lasagne.layers.EmbeddingLayer(
        l_inp,
        input_size=vocab_size,  # size of embedding = number of words
        output_size=embedding_size,  # vector size used to represent each word
        W=ini)

    if dropout_frac > 0:
        l_emb = lasagne.layers.DropoutLayer(l_emb, p=dropout_frac)

    # first layer
    l_rec1 = lasagne.layers.LSTMLayer(
        l_emb,
        num_units=rec_num_units,
        peepholes=False,
        W_hid_to_cell=ini,
        W_in_to_cell=ini,
        W_hid_to_forgetgate=ini,
        W_in_to_forgetgate=ini,
        W_hid_to_ingate=ini,
        W_in_to_ingate=ini,
        W_hid_to_outgate=ini,
        W_in_to_outgate=ini)

    if dropout_frac > 0:
        l_rec1 =  lasagne.layers.DropoutLayer(l_rec1, p=dropout_frac)

    l_rec2 = lasagne.layers.LSTMLayer(
        l_rec1,
        num_units=rec_num_units,
        peepholes=False,
        W_hid_to_cell=ini,
        W_in_to_cell=ini,
        W_hid_to_forgetgate=ini,
        W_in_to_forgetgate=ini,
        W_hid_to_ingate=ini,
        W_in_to_ingate=ini,
        W_hid_to_outgate=ini,
        W_in_to_outgate=ini)
    # output_network=output_network)

    l_shp = lasagne.layers.ReshapeLayer(l_rec2,
                                        (batch_size*model_seq_len, rec_num_units))
    l_out = lasagne.layers.DenseLayer(l_shp,
                                      num_units=vocab_size,
                                      nonlinearity=lasagne.nonlinearities.softmax)
    l_out = lasagne.layers.ReshapeLayer(l_out,
                                        (batch_size, model_seq_len, vocab_size))
    return l_out

# Define symbolic theano variables

l_out = build_model(BATCH_SIZE, MODEL_SEQ_LEN)
#l_out_test = build_model(1, x_test.shape[0]-1)
sym_x = T.imatrix()


def calc_cross_ent(net_output):
    """
    TODO

    :param net_output:
    :return:
    """
    preds = net_output[:, :-1]
    targets = sym_x[:, 1:]

    # we need to flatten x_preds and x_targets
    preds = T.reshape(preds, (BATCH_SIZE * (MODEL_SEQ_LEN - 1), vocab_size))
    preds += TOL  # add constant for numerical stability
    targets = T.flatten(targets)

    cost = T.nnet.categorical_crossentropy(preds, targets)

    return cost, preds

# note the use of deterministic keyword to disable dropout during eval
train_out = lasagne.layers.get_output(l_out, sym_x, deterministic=False)
eval_out = lasagne.layers.get_output(l_out, sym_x, deterministic=True)
#test_out = lasagne.layers.get_output(l_out_test, sym_x, deterministic=True)

cost_temp, preds_train = calc_cross_ent(train_out)
cost_train = T.mean(cost_temp)
cost_train_sum = T.sum(cost_temp)

cost_eval = T.sum(calc_cross_ent(train_out)[0])

# given a list of parameters and a cost function theano will automatiacally
# calculate the gradients
all_params = lasagne.layers.get_all_params(l_out, trainable=True)
all_grads = T.grad(cost_train, all_params)

# Wit the gradients for each parameter we can calcualte update rules for each
# parameter. Lasagne implements a number of update rules, here we'll use
# Adam and step_scaling
all_grads, norm, multiplier = lasagne.updates.step_scaling(all_grads, 5.0)
updates = lasagne.updates.adam(all_grads, all_params, learning_rate=lr)

# define training function. The update arg specifies that the parameters
# should be updated using the adam updates.
print("compiling f_train...")
f_train = theano.function(
    [sym_x], [cost_train, cost_train_sum, norm], updates=updates)

# Define eval function.
print("compiling f_eval...")
f_eval = theano.function([sym_x], [cost_eval], )

def calc_perplexity(x):
    """
    TODO: Write expression for plexity, figure out why
    perp_log_sum / n_samples is probably equal to the n_sample'th root


    THIS CUTS OF A BATCH, DONT USE IT FOR FINAL PERFORMANCE!
    :param x:
    :param mask:
    :return:
    """

    n_batches = x.shape[0] // BATCH_SIZE
    l_cost = []
    for i in range(n_batches):
        x_batch = x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]   # single batch
        l_cost.append(f_eval(x_batch)[0])

    # sum because we miss one target at eveary sample

    l = BATCH_SIZE*(MODEL_SEQ_LEN-1)*n_batches
    perplexity = np.exp(np.sum(l_cost) / l)

    return perplexity

n_batches_train = x_train.shape[0] // BATCH_SIZE   # floor
bs = BATCH_SIZE
batches = [range(j * bs, (j + 1) * bs) for j in range(n_batches_train)]

for epoch in range(num_epochs):
    idx = np.random.choice(x_train.shape[0], n_batches_train*bs, replace=True)

    l_cost, l_norm = [], []

    batch_time = time.time()
    for i in range(n_batches_train):
        this_batch = batches[i]
        this_indices = idx[this_batch]
        x_batch = x_train[this_indices]   # single batch
        cost, cost_sum, norm = f_train(x_batch)
        l_cost.append(cost_sum)
        l_norm.append(norm)

        if (i+1) % eval_freq == 0 or i+1 == n_batches_train:
            elapsed = time.time() - batch_time
            words_per_second = float(BATCH_SIZE*(MODEL_SEQ_LEN)*len(l_cost)) / elapsed
            n_words_evaluated = BATCH_SIZE*(MODEL_SEQ_LEN-1)*len(l_cost)
            perplexity_valid = calc_perplexity(x_valid)
            perplexity_train = np.exp(np.sum(l_cost) / n_words_evaluated)
            print("Epoch           :", float(len(l_cost)) / n_batches_train)
            print("Perplexity Train:", perplexity_train)
            print("Perplexity valid:", perplexity_valid)
            print("Words per second:", words_per_second)
            l_cost = []
            batch_time = 0
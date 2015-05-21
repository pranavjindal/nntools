from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import os
import time
import sys
import lasagne


#  SETTINGS
folder = 'penntree'
BATCH_SIZE = 20         # batch size
MODEL_SEQ_LEN = 20      # how many steps to unroll
LARGE_MODEL = False
TOL = 1e-6              # numerial stability

if LARGE_MODEL:
    ini = lasagne.init.Uniform(0.04)
    REC_NUM_UNITS = 1500
    embedding_size = 200
    dropout_frac = 0.65
    lr = 1.0
    decay = 1.15
    max_grad_norm = 10
    num_epochs = 55
    no_decay_epochs = 14
else:
    ini = lasagne.init.Uniform(0.1)
    REC_NUM_UNITS = 200
    embedding_size = REC_NUM_UNITS
    dropout_frac = 0
    lr = 1.0
    decay = 2.0
    max_grad_norm = 10
    num_epochs = 1000
    no_decay_epochs = 5


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
    Note that the function has the side effects that vocab_map and vocab_idx are
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


def reorder(x, batch_size, model_seq_len):
    """
    Rearranges dataset so batehes process sequential data.

    The data is reordered such that position n in each batch corresponds to a
    sequence.

    some_sequence = batch_1[n] + batch_1[n] ... batch_j[n]

    Parameters
    ----------
    x : 1D numpy.array
    batch_size : int
    model_seq_len : int
        number of steps the model is unrolled

    Returns
    -------
    Reoredered x and reoredered targets. Targets are shifted version of x

    """
    if x.ndim != 1:
        raise ValueError("Data must be 1D, was", x.ndim)

    if x.shape[0] % (batch_size*model_seq_len) == 0:
        print(" x.shape[0] % (batch_size*model_seq_len) == 0 -> x is "
              "set to x = x[:-1]")
        x = x[:-1]

    x_resize =  \
        (x.shape[0] // (batch_size*model_seq_len))*model_seq_len*batch_size
    n_samples = x_resize // (model_seq_len)
    n_batches = n_samples // batch_size

    targets = x[1:x_resize+1].reshape(n_samples, model_seq_len)
    x_out = x[:x_resize].reshape(n_samples, model_seq_len)

    out = np.zeros(n_samples, dtype=int)
    for i in range(n_batches):
        val = range(i, n_batches*batch_size+i, n_batches)
        out[i*batch_size:(i+1)*batch_size] = val

    x_out = x_out[out]
    targets = targets[out]

    return x_out.astype('int32'), targets.astype('int32')


def traindata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data(os.path.join(folder, "ptb.train.txt"),
                  vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)


def validdata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data(os.path.join(folder, "ptb.valid.txt"),
                  vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)


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


def testdata(model_seq_len, batch_size, vocab_map, vocab_idx):
    return load_data_masked(
        model_seq_len, batch_size, vocab_map, vocab_idx, "ptb.test.txt")


# vocab_map and vocab_idx are updated as side effects of load_data
vocab_map = {}
vocab_idx = [0]
x_train, y_train = traindata(MODEL_SEQ_LEN, BATCH_SIZE, vocab_map, vocab_idx)
x_test, mask_test = testdata(MODEL_SEQ_LEN, BATCH_SIZE, vocab_map, vocab_idx)
x_valid, y_valid = validdata(MODEL_SEQ_LEN, BATCH_SIZE, vocab_map, vocab_idx)
vocab_size = vocab_idx[0]
eval_freq = 10000000   # number of batches between eval


print("-" * 80)
print("Vocab size:s", vocab_size)
print("Data shapes")
print("Train data:", x_train.shape)
print("Test data :", x_test.shape)
print("Valid data:", x_valid.shape)
print("-" * 80)

# Theno symbolic vars
sym_x = T.imatrix()
sym_y = T.imatrix()

cell1_init_sym = T.matrix()
hid1_init_sym = T.matrix()
cell2_init_sym = T.matrix()
hid2_init_sym = T.matrix()


# BUILDING THE MODEL
# ------------------
# For a languge model we want to use the previous prediction as input in t+1.
# In Lasagne you do that by giving an output network in the output_network arg.
#
# Model structure:
#
#    embedding -> LSTM1 --> LSTM2 --> output network------> predictions
l_inp = lasagne.layers.InputLayer((BATCH_SIZE, MODEL_SEQ_LEN))

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
    num_units=REC_NUM_UNITS,
    peepholes=False,
    W_hid_to_cell=ini,
    W_in_to_cell=ini,
    W_hid_to_forgetgate=ini,
    W_in_to_forgetgate=ini,
    W_hid_to_ingate=ini,
    W_in_to_ingate=ini,
    W_hid_to_outgate=ini,
    W_in_to_outgate=ini,
    learn_init=False,
    hid_init_val=[cell1_init_sym, hid1_init_sym])

if dropout_frac > 0:
    l_drp1 = lasagne.layers.DropoutLayer(l_rec1, p=dropout_frac)
else:
    l_drp1 = l_rec1

l_rec2 = lasagne.layers.LSTMLayer(
    l_rec1,
    num_units=REC_NUM_UNITS,
    peepholes=False,
    W_hid_to_cell=ini,
    W_in_to_cell=ini,
    W_hid_to_forgetgate=ini,
    W_in_to_forgetgate=ini,
    W_hid_to_ingate=ini,
    W_in_to_ingate=ini,
    W_hid_to_outgate=ini,
    W_in_to_outgate=ini,
    learn_init=False,
    hid_init_val=[cell2_init_sym, hid2_init_sym])
# output_network=output_network)

if dropout_frac > 0:
    l_drp2 = lasagne.layers.DropoutLayer(l_rec2, p=dropout_frac)
else:
    l_drp2 = l_rec2

l_shp = lasagne.layers.ReshapeLayer(l_drp2,
                                    (BATCH_SIZE*MODEL_SEQ_LEN, REC_NUM_UNITS))
l_out = lasagne.layers.DenseLayer(l_shp,
                                  num_units=vocab_size,
                                  nonlinearity=lasagne.nonlinearities.softmax)
l_out = lasagne.layers.ReshapeLayer(l_out,
                                    (BATCH_SIZE, MODEL_SEQ_LEN, vocab_size))


def calc_cross_ent(net_output, targets):
    preds = T.reshape(net_output, (BATCH_SIZE * MODEL_SEQ_LEN, vocab_size))
    preds += TOL  # add constant for numerical stability
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)
    return cost

# note the use of deterministic keyword to disable dropout during eval
train_out = lasagne.layers.get_output(l_out, sym_x, deterministic=False)
hidden_states_train = [l_rec1.cell, l_rec1.hid, l_rec2.cell, l_rec2.hid]

eval_out = lasagne.layers.get_output(l_out, sym_x, deterministic=True)
hidden_states_eval = [l_rec1.cell, l_rec1.hid, l_rec2.cell, l_rec2.hid]


cost_train = T.mean(calc_cross_ent(train_out, sym_y))
cost_eval = T.mean(calc_cross_ent(eval_out, sym_y))


all_params = lasagne.layers.get_all_params(l_out, trainable=True)

# Calculate gradients w.r.t cost function. Note that we scale the cost with
# MODEL_SEQ_LEN. This is to be consistent with
# https://github.com/wojzaremba/lstm . The scaling is due to difference
# between torch and theano.
# We could have also scaled the learning rate, and also rescaled the
# norm constraint.
all_grads = T.grad(cost_train*MODEL_SEQ_LEN, all_params)

# With the gradients for each parameter we can calculate update rules for each
# parameter. Lasagne implements a number of update rules, here we'll use
# sgd and step_scaling
all_grads, norm, multiplier = lasagne.updates.step_scaling(
    all_grads, max_grad_norm)

# Use shared variable for learning rate. Allows us to change the learning rate
# during training.
sh_lr = theano.shared(lasagne.utils.floatX(lr))
updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=sh_lr)

# define training function. The update arg specifies that the parameters
# should be updated using the update rules.
print("compiling f_eval...")
fun_inp = [sym_x, sym_y,
           cell1_init_sym, hid1_init_sym, cell2_init_sym, hid2_init_sym]
f_eval = theano.function(fun_inp,
                         [cost_eval,
                          hidden_states_eval[0][:, -1],
                          hidden_states_eval[1][:, -1],
                          hidden_states_eval[2][:, -1],
                          hidden_states_eval[3][:, -1]])

print("compiling f_train...")
f_train = theano.function(fun_inp,
                          [cost_train,
                           norm,
                           hidden_states_train[0][:, -1],
                           hidden_states_train[1][:, -1],
                           hidden_states_train[2][:, -1],
                           hidden_states_train[3][:, -1]],
                          updates=updates)


def calc_perplexity(x, y):
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
    cell1, hid1, cell2, hid2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                                         dtype='float32') for _ in range(4)]

    for i in range(n_batches):
        x_batch = x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]   # single batch
        y_batch = y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]   # single batch
        cost, cell1, hid1, cell2, hid2 = f_eval(
            x_batch, y_batch, cell1, hid1, cell2, hid2)
        l_cost.append(cost)

    n_words_evaluated = (x.shape[0] - 1) / MODEL_SEQ_LEN
    perplexity = np.exp(np.sum(l_cost) / n_words_evaluated)

    return perplexity

n_batches_train = x_train.shape[0] // BATCH_SIZE   # floor
for epoch in range(num_epochs):
    l_cost, l_norm = [], []

    batch_time = time.time()
    cell1, hid1, cell2, hid2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                                         dtype='float32') for _ in range(4)]
    for i in range(n_batches_train):
        x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]   # single batch
        y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        cost, norm, cell1, hid1, cell2, hid2 = f_train(
            x_batch, y_batch, cell1, hid1, cell2, hid2)
        l_cost.append(cost)
        l_norm.append(norm)

    if epoch > (no_decay_epochs - 1):
        current_lr = sh_lr.get_value()
        sh_lr.set_value(T.cast(current_lr / float(decay), 'float32'))

    elapsed = time.time() - batch_time
    words_per_second = float(BATCH_SIZE*(MODEL_SEQ_LEN)*len(l_cost)) / elapsed
    n_words_evaluated = (x_train.shape[0] - 1) / MODEL_SEQ_LEN
    perplexity_valid = calc_perplexity(x_valid, y_valid)
    perplexity_train = np.exp(np.sum(l_cost) / n_words_evaluated)
    print("Epoch           :", epoch)
    print("Perplexity Train:", perplexity_train)
    print("Perplexity valid:", perplexity_valid)
    print("Words per second:", words_per_second)
    l_cost = []
    batch_time = 0
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for profiling to sync gpu calls disable for full run
import os.path

#import scipy.io
import lasagne    # nn packages for layers nn layers + lstm
import theano
import scipy.io

theano.config.allow_gc=False
theano.scan.allow_gc=False
#theano.config.profile=True

#theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'
#theano.config.mode = 'DEBUG_MODE'
theano.config.compute_test_value = 'raise'
#theano.config.optimizer = None
#theano.config.exception_verbosity='high'


import theano.tensor as T
import numpy as np
import time
from ConfusionMatrix import ConfusionMatrix          # confusion matrix class, ported from torch
from LSTMTrainingFunctions import savemodel
from LSTMTrainingFunctions import loadmodel
from LSTMTrainingFunctions import adadelta_normscaled
from LSTMTrainingFunctions import nesterov_normscaled
from LSTMTrainingFunctions import padtobatchmultiplesimple
from LSTMTrainingFunctions import createmodel
from training_data_funcs import *


import logging
np.random.seed(1234)
BATCH_SIZE = 64
N_EPOCHS =  100
VERBOSE = False
N_HID = 100


#####################################################################################
#  LOAD DATA                                                                        #
#####################################################################################
# the code expect X data to be shape (batch_size x seqlen x nfeatures)
# targets to be (batch_size x seqlen x nclasses), i.e one hot encoded classes
# and the mask to be (batch_size x seqlen x 1)
#
#
# The theano graph is compiled to only accept a specific batch_size, and
# a specific sequence length.
# To handle a specific sequence length all sequences are padded to be equal
# length. A mask is used to indicate if the sequence was padded or not.
# The mask is 1 when the sequence is not padded and 0 when the sequence is
# padded.
#
# Simiarly theano graphs only accept batches of a specified size. We therefore
# need to bad the number of sequences to be a multiple of batch size.
# This is performed with the function padtomultiplesimple.
# For training data we randomly select sequences so this type of padding is
# only needed for validation and test sets.
TRAIN_NC = '../data/train_1_speaker.nc'
VAL_NC = '../data/val_1_speaker.nc'

X_train, y_train = load_netcdf(TRAIN_NC)
X_train = X_train
y_train = y_train
X_val, y_val = load_netcdf(VAL_NC)
X_val = X_val
y_val = y_val

# Find the longest sequence
MAX_SEQ_LENGTH = max(max([X.shape[0] for X in X_train]),
             max([X.shape[0] for X in X_val]))
# Convert to batches of time series of uniform length
X_train, _ = make_batches(X_train, MAX_SEQ_LENGTH)
y_train, mask_train = make_batches(y_train, MAX_SEQ_LENGTH)
X_val, _ = make_batches(X_val, MAX_SEQ_LENGTH)
y_val, mask_val = make_batches(y_val, MAX_SEQ_LENGTH)


N_FEATURES = X_train.shape[-1]
N_CLASSES = y_train.shape[-1]
# reshape to format batches x max_seq_len x n_features
X_train = X_train.reshape((-1, MAX_SEQ_LENGTH, N_FEATURES))
y_train = y_train.reshape((-1, MAX_SEQ_LENGTH, N_CLASSES))
mask_train = mask_train[:, :, :, 0].reshape((-1, MAX_SEQ_LENGTH, 1))

X_val = X_val.reshape((-1, MAX_SEQ_LENGTH, N_FEATURES))
y_val = y_val.reshape((-1, MAX_SEQ_LENGTH, N_CLASSES))
mask_val = mask_val[:, :, :, 0].reshape((-1, MAX_SEQ_LENGTH, 1))


N_SAMPLES_TRAIN_PADDED = X_train.shape[0]
N_SAMPLES_VAL_PADDED = X_val.shape[0]

X_val, y_val, mask_val = padtobatchmultiplesimple(X_val, y_val, mask_val, BATCH_SIZE)
X_train, y_train, mask_train = padtobatchmultiplesimple(X_train, y_train, mask_train, BATCH_SIZE)

y_train = np.argmax(y_train, axis=-1).astype(theano.config.floatX)
y_val = np.argmax(y_val, axis=-1).astype(theano.config.floatX)

#######################################################
# symbolic variables                                  #
#######################################################
# Theano defines its computations using symbolic variables. A symbolic variable
# is a matrix, vector, 3D matrix and specifies the data type.
# A symbolic value does not hold any data, like a matlab matrix or np.array
# Note that mask is contstructed with a broadcastable argument which specifies
# that the mask can be broadcasted in the 3. dimension.

sym_input = T.tensor3('input')   # float32
sym_mask = T.TensorType(dtype=theano.config.floatX,
                        broadcastable=(False,False,True))('mask')
sym_target = T.imatrix('target_output')  # integer symbolic variable

#######################################################
# SHARED VARIABLES                                    #
#######################################################
# Data is loaded on the GPU using theano.shared_variables.
# I use the naming convention that sh_SOMETHING will be used for the values
# for the symbolic variable sym_SOMETHING
batch_size_X = (BATCH_SIZE, MAX_SEQ_LENGTH, N_FEATURES)
batch_size_y = (BATCH_SIZE, MAX_SEQ_LENGTH, N_CLASSES)

sh_input = theano.shared(
    np.zeros(shape=batch_size_X, dtype=theano.config.floatX), borrow=True)


sh_target = theano.shared(
        np.zeros(shape=(BATCH_SIZE, MAX_SEQ_LENGTH),
                 dtype=theano.config.floatX), borrow=True)

sh_mask = theano.shared(np.zeros(shape=(BATCH_SIZE, MAX_SEQ_LENGTH, 1),
                dtype=theano.config.floatX), borrow=True,
                broadcastable=(False, False, True)) #<---IMPORTANT Allows to multiply m x n x 1 matrix with m x n x z mat

# testing values
sym_target.tag.test_value = \
    (np.random.rand(BATCH_SIZE, MAX_SEQ_LENGTH)>0.3).astype('int32')
sym_mask.tag.test_value = \
    np.random.rand(BATCH_SIZE, MAX_SEQ_LENGTH, 1).astype(theano.config.floatX)
sym_input.tag.test_value = \
    np.random.rand(*batch_size_X).astype(theano.config.floatX)


####################
# Setup Bidirectional LSTM
####################
manual = True
if manual:
    peepholes = False
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_SEQ_LENGTH, N_FEATURES))
    #l_in = lasagne.layers.GaussianNoiseLayer(l_in, sigma=0.6)
    recout = lasagne.layers.BidirectionalLSTMLayer(
        l_in, num_units=3, dropout_rate=0.0, peepholes=peepholes, learn_init=False)
    recout = lasagne.layers.BidirectionalLSTMLayer(
        recout, num_units=4, dropout_rate=0.0, peepholes=peepholes, learn_init=False)
    recout = lasagne.layers.BidirectionalLSTMLayer(
        recout, num_units=5, dropout_rate=0.0, peepholes=peepholes, learn_init=False)
    l_reshape = lasagne.layers.ReshapeLayer(
        recout,  (BATCH_SIZE*MAX_SEQ_LENGTH, recout.get_output_shape()[-1]))
    l_reshape = lasagne.layers.DenseLayer(
        l_reshape, num_units=7, nonlinearity=lasagne.nonlinearities.rectify)
    l_rec_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=N_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
    l_out = lasagne.layers.ReshapeLayer(
        l_rec_out, (BATCH_SIZE, MAX_SEQ_LENGTH, N_CLASSES))
else:
    l_out, _, l_in = createmodel(rnn_layer_layers=[3,4,5],
                         isbrnn=True,
                         batch_size=BATCH_SIZE,
                         n_features=N_FEATURES,
                         n_classes=N_CLASSES,
                         layer_type_rnn="LSTMFAST",
                         padded_seq_len=MAX_SEQ_LENGTH,
                         output_layers=[7],
                         input_layers=None,
                         learn_init=True,
                         final_output_layer="softmax")


# createnet in LSTMTrainingFunctions can setup networks with a number of
# different architectures.

# Cross entropy cost function.
# Note that we use the mask to ignore masked sequences during cost calculation
def costfun(p_y_given_x, y, mask,db='COST:'):
    shape = (BATCH_SIZE*MAX_SEQ_LENGTH, N_CLASSES)
    y_reshape = y.flatten()
    pyx_reshape = p_y_given_x.reshape(shape)
    mask_reshape = mask.flatten()
    xe_unnorm = -T.log(
        pyx_reshape[T.arange(BATCH_SIZE*MAX_SEQ_LENGTH), y_reshape]+ 1e-8)
    xe = T.sum( xe_unnorm * mask_reshape) / T.sum(mask)

    if VERBOSE:   #DEBUGING
        xe = theano.printing.Print(db)(xe)
    return xe

##################################################################
#           THEANO FUNCTIONS                                     #
##################################################################
# specify that the input to the network is l_in and that the input symbol
# is input_sym. If we have several input layers you can specify more key:value
# pairs
input_dict = {l_in: sym_input}

# create cost entropy costfunctions.
# We use the get_output method to get the output from the network.
# When you use dropout layers you shuld set deterministic to false during
# training and to true during testing. In theano this requires two different
# graphs.
# When we use backwards LSTM's a symbolic variable representing mask must be
# given as argument.
print "CREATING COST FUNCTIONS...",
cost_train = costfun(
    l_out.get_output(input_dict, deterministic=False,mask=sym_mask),
    sym_target, sym_mask, db='COST TRAIN:')
cost_val = costfun(
    l_out.get_output(input_dict, deterministic=True,mask=sym_mask),
    sym_target, sym_mask, db='COST VAL:')
print "DONE"


# Get a list of all parameters in the network
all_params = lasagne.layers.get_all_params(l_out)

# Given a cost function (cost_train) and a list of parameters Theano can
# calculate the gradients and update rules w.r.t to each parameter.
# We use adadelta, which automatically tunes the learning rate.
# adadelta_normscaled returns a list of update rules for each parameter
#updates = adadelta_normscaled(
#    cost_train, all_params,batch_size=BATCH_SIZE,learning_rate=1.0,
#    epsilon=10e-6, max_norm=0.02, verbose=VERBOSE)

print "CALCULATING UPDATES...",
updates = nesterov_normscaled( cost_train, all_params, 0.01, 0.5, BATCH_SIZE)
print "DONE"

# print number of params
total_params = sum([p.get_value().size for p in all_params])
print "#NETWORK params:", total_params


# These lists specify that sym_input should take the value of sh_input and etc.
# Note the cast: T.cast(sh_target, 'int32'). This is nessesary because Theano
# does only support shared varibles with type float32. We cast the shared
# value to an integer before it is used in the graph.
givens = [(sym_input, sh_input),
          (sym_target, T.cast(sh_target, 'int32')),
         (sym_mask, sh_mask)]
givens_preds = [(sym_input, sh_input), (sym_mask, sh_mask)]

# theano.function compiles a theano graph. [] means that the the function
# takes no input because the inputs are specified with the givens argument.
# We compile cost_train and specify that the parameters should be updated
# using the adadelta update rules.
print "COMPILING FUNCTIONS...TRAIN",
train = theano.function([], cost_train, updates=updates, givens=givens)
print "...COST-VAL...",
compute_cost_val = theano.function([], cost_val, givens=givens)
print "...preds..."
compute_preds = theano.function([], l_out.get_output(input_dict, determinist=True, mask=sym_mask),
                                  givens=givens_preds)
print "Done"
####################################################################
#           TRAINING LOOP                                          #
####################################################################
confmatrix = ConfusionMatrix(N_CLASSES)
n_batches_train = int(np.ceil(float(X_train.shape[0]) / BATCH_SIZE))
n_batches_val = int(np.ceil(float(X_val.shape[0]) / BATCH_SIZE))
seq_count = 0
best_acc_val = 0
for epoch in range(N_EPOCHS):
    # single epoch training
    seq_shuffle = np.random.choice(N_SAMPLES_TRAIN_PADDED, N_SAMPLES_TRAIN_PADDED, False)
    batches_train = []

    # construct traning batches
    for i in range(n_batches_train):
        seqs_in_batch = []
        for j in range(BATCH_SIZE):
            if seq_count > N_SAMPLES_TRAIN_PADDED - 1:
                seq_shuffle = np.random.choice(N_SAMPLES_TRAIN_PADDED, N_SAMPLES_TRAIN_PADDED, False)
                seq_count = 0
            seqs_in_batch.append(seq_shuffle[seq_count % N_SAMPLES_TRAIN_PADDED])
            seq_count += 1
        batches_train.append(seqs_in_batch)

    start_time = time.time()
    c = 0
    for ind in batches_train:
        c += 1
        if c % 5 == 0:
            print '%i ' % c,
        # update shared variables

        # extract training batches
        y_train_batch = y_train[ind, ]
        X_train_batch = X_train[ind, ]
        train_mask_batch = mask_train[ind, ]

        # update value of shared variables to current mini batch
        sh_input.set_value(X_train_batch, borrow=True)
        sh_target.set_value(y_train_batch, borrow=True)
        sh_mask.set_value(train_mask_batch, borrow=True)

        train()
        if VERBOSE:
            print ""
            train_preds_probs = compute_preds()
            train_mask_flat = train_mask_batch.flatten()
            train_model_preds_lab = np.argmax(train_preds_probs, axis=-1).flatten()
            train_true_preds_lab = y_train[ind, ].flatten()
            confmatrix.batchAdd(train_true_preds_lab[train_mask_flat],
                                train_model_preds_lab[train_mask_flat])
            print '------------->', confmatrix.accuracy()
            print confmatrix
            confmatrix.zero()

    end_time = time.time()

    if VERBOSE:
        print '#' * 20 + 'TRAIN CONFUSION' + '#' * 20
        print confmatrix
        confmatrix.zero()
        print '#' * 55

    # VALIDATION
    batches_val = [range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
                   for i in range(n_batches_val)]
    val_cost = 0
    all_val_masks, all_val_model_preds, all_val_true_preds = [], [], []
    for ind in batches_val:
        # update shared variables
        y_val_batch = y_val[ind, ]
        X_val_batch = X_val[ind,]
        val_mask_batch = mask_val[ind, ]

        # update value of shared variables to current mini batch
        sh_input.set_value(X_val_batch, borrow=True)
        sh_target.set_value(y_val_batch, borrow=True)
        sh_mask.set_value(val_mask_batch, borrow=True)

        val_cost += compute_cost_val()
        val_preds_probs = compute_preds()
        val_model_preds_lab = np.argmax(val_preds_probs, axis=-1)
        val_true_preds_lab = y_val[ind, ]

        # val_mask_batch is shape (batchsize x seqlen x 1)
        # model_preds_lab and true_preds_lab are shape (batchsize x seqlen)
        # because they are the same size we can flatten mask, preds and true
        # then use mask_flat to extract the non-masked samples and lastly
        # add everything to the confusionmatrix.
        # if assert does not hold, flatten will not produce correct results
        assert val_model_preds_lab.shape == val_mask_batch.squeeze().shape
        assert val_true_preds_lab.shape == val_mask_batch.squeeze().shape

        val_mask_flat = val_mask_batch.flatten()
        val_model_preds_lab_flat = val_model_preds_lab.flatten()[val_mask_flat]
        val_true_preds_lab_flat = val_true_preds_lab.flatten()[val_mask_flat]
        confmatrix.batchAdd(val_true_preds_lab_flat, val_model_preds_lab_flat)

        all_val_masks.append(val_mask_batch.squeeze(axis=-1))
        all_val_model_preds.append(val_model_preds_lab)
        all_val_true_preds.append(val_true_preds_lab)

    val_cost /= n_batches_val


    current_acc_val = confmatrix.accuracy()

    print ""
    print '------ VAL ACC ------>', current_acc_val
    print '------ VAL COST ----->', val_cost
    print confmatrix
    assert np.sum(mask_val) == np.sum(confmatrix.getMat())
    confmatrix.zero()


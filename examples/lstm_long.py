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
#theano.config.compute_test_value = 'raise'
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
import argparse
import csv
#np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser()
#parser.add_argument("-test", type=str, help="Test Split")
parser.add_argument("-rnnunit", type=str, help="unit type tanh|sigm|ReLU|LAunit", default='tanh')
parser.add_argument("-outfolder", type=str, help="Folder where logfile and predictions are saved", default='../logs')
parser.add_argument("-output", type=str, help="output layer type linear or softmax", default='softmax')
parser.add_argument("-rnnlayers", type=str, help="size of rnn layers", default='100,100')
parser.add_argument("-rnnlayerdropout", type=float, help="dropout rnn layer", default=0.2)
parser.add_argument("-outputlayers", type=str, help="size of output layers, EMPTY is no layers", default='200,200')
parser.add_argument("-outputlayerdropout", type=float, help="output layer dropout", default=0.5)
parser.add_argument("-inputlayers", type=str, help="size of input layers, EMPTY is no layers", default ='EMPTY')
parser.add_argument("-inputlayerdropout", type=float, help="input layer dropout", default=0.5)
parser.add_argument("-rnnlayertype", type=str,
                    help="'LSTMFAST'|'LSTMnopeepsFAST'|'RNN'",
                    default='LSTMnopeepsFAST')
parser.add_argument("-batchsize", type=int, help="batchsize", default=16)
parser.add_argument("-epochs", type=int, help="number of epochs", default=100)
parser.add_argument("-verbose", type=str, help="verbose output", default='False')
parser.add_argument("-maxgradnorm", type=float, help="max gradient of updates", default=0.1)
parser.add_argument("-relucap", type=float, help="capping of ReLU", default=0)
parser.add_argument("-reluleakyness", type=float, help="leakyness of ReLU [0-1]", default=0)
parser.add_argument("-loadnetwork", type=str, help="restore previous network", default='False')
parser.add_argument("-LAunitL2", type=float, help="L2 reg on LAunits", default=0.001)
parser.add_argument("-timebatches", type=str, help="prints processing time for single batches", default='False')
parser.add_argument("-inputwindow", type=int, help="input window size", default=1)
args = parser.parse_args()

print  "#"*80
for name,val in sorted(vars(args).items()):
    print name," "*(40-len(name)),val
print  "#"*80

def parselayer(arg):
    if arg == 'EMPTY':
        return False
    else:
        return map(int, arg.split(','))

np.random.seed(1234)
##############################
# Settings
##############################
MAX_SEQ_LENGTH = 216

OUTPUT = args.output

# number of gated layers
RNN_HIDDEN_LAYERS = parselayer(args.rnnlayers)
RNN_DROPOUT_RATE = float(args.rnnlayerdropout)
RNN_IS_BIDIRECTIONAL = True
RNN_LAYER_TYPE = args.rnnlayertype

# add layer or merge network
OUTPUT_LAYERS =parselayer(args.outputlayers)
OUTPUT_DROPOUT_RATE = args.outputlayerdropout
INPUT_LAYERS = parselayer(args.inputlayers)
INPUT_DROPOUT_RATE = args.inputlayerdropout

VERBOSE = True if args.verbose=='True' else False
PRINT_BATCH_TIMINGS = True if args.timebatches=='True' else False
LEARN_INIT = True
RELU_CAP = args.relucap
RELU_LEAKYNESS = args.reluleakyness
LAUNITL2 = args.LAunitL2
UNITTYPE_RNN = args.rnnunit
MAX_NORM_GRADIENTS = args.maxgradnorm  # crashes if not float


BATCH_SIZE = args.batchsize
N_EPOCHS =  args.epochs
LOGFOLDER = args.outfolder


### Create folders
outfolder = LOGFOLDER
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

filename_base = "Train"
FILE_PICKLE = outfolder + "/" + filename_base + ".bestmodelpickle"
LOAD_NETWORK = True if args.loadnetwork=='True' else False
if LOAD_NETWORK == True:
    print "LOADING NETOWRK", FILE_PICKLE
else:
    print "Starting NEW training"


logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(outfolder + "/" + filename_base + ".log",mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

#####################################################################################
#  SETUP TRAINING AND VALIDATION DATA / TEST DATA IS CB513 NOT LOADED               #
#####################################################################################
TRAIN_NC = '../data/train_1_speaker.nc'
VAL_NC = '../data/val_1_speaker.nc'

logger.info('Loading data...')
X_train, y_train = load_netcdf(TRAIN_NC)
X_train = X_train
y_train = y_train
X_val, y_val = load_netcdf(VAL_NC)
X_val = X_val
y_val = y_val

# Find the longest sequence
length = max(max([X.shape[0] for X in X_train]),
             max([X.shape[0] for X in X_val]))
# Convert to batches of time series of uniform length
X_train, _ = make_batches(X_train, length)
y_train, mask_train = make_batches(y_train, length)
X_val, _ = make_batches(X_val, length)
y_val, mask_val = make_batches(y_val, length)

# the code expect X data to be shape (batch_size x seqlen x nfeatures
# targets to be (batch_size x seqlen x nclasses), i.e one hot encoded classes
# and the mask to be (batch_size x seqlen x 1)
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

#######################################################
# symbolic variables                                  #
#######################################################
input = T.tensor3('input')
mask = T.TensorType(dtype=theano.config.floatX, broadcastable=(False,False,True))('mask')
target_output = T.imatrix('target_output')

#######################################################
# SHARED VARIABLES                                    #
#######################################################
batch_size_X = (BATCH_SIZE, MAX_SEQ_LENGTH, N_FEATURES)
batch_size_y = (BATCH_SIZE, MAX_SEQ_LENGTH, N_CLASSES)

sh_input = theano.shared(np.zeros(shape=batch_size_X, dtype=theano.config.floatX), borrow=True)
input.tag.test_value = np.random.rand(*batch_size_X).astype(theano.config.floatX)


sh_target_output = theano.shared(
        np.zeros(shape=(BATCH_SIZE, MAX_SEQ_LENGTH),
                 dtype=theano.config.floatX), borrow=True)
target_output.tag.test_value = (np.random.rand(BATCH_SIZE, MAX_SEQ_LENGTH)>0.3).astype('int32')
sh_mask = theano.shared(np.zeros(shape=(BATCH_SIZE, MAX_SEQ_LENGTH, 1),
                dtype=theano.config.floatX), borrow=True,
                broadcastable=(False, False, True)) #<---IMPORTANT Allows to multiply m x n x 1 matrix with m x n x z mat
mask.tag.test_value = np.random.rand(BATCH_SIZE, MAX_SEQ_LENGTH, 1).astype(theano.config.floatX)                                   #

def costfun(p_y_given_x, y, mask,db='COST:'):
    n_samples = BATCH_SIZE*MAX_SEQ_LENGTH
    shape = (n_samples, N_CLASSES)
    y_reshape = y.flatten()
    pyx_reshape = p_y_given_x.reshape(shape)
    mask_reshape = mask.flatten()
    xe_unnorm = -T.log(pyx_reshape)[T.arange(n_samples), y_reshape]
    xe = T.sum( xe_unnorm * mask_reshape) / T.sum(mask)

    if VERBOSE:   #DEBUGING
        xe = theano.printing.Print(db)(xe)
    return xe


def createnet():
    return createmodel(
        rnn_layer_layers=RNN_HIDDEN_LAYERS,
        isbrnn=RNN_IS_BIDIRECTIONAL,
        batch_size=BATCH_SIZE,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        layer_type_rnn=RNN_LAYER_TYPE,
        padded_seq_len=MAX_SEQ_LENGTH,
        input_layers=INPUT_LAYERS,
        input_layer_dropout=INPUT_DROPOUT_RATE,
        output_layers=OUTPUT_LAYERS,
        final_output_layer=OUTPUT,
        learn_init=LEARN_INIT,
        dropout_rnn=RNN_DROPOUT_RATE,
        output_layer_dropout=OUTPUT_DROPOUT_RATE,
        unittype_rnn=UNITTYPE_RNN,
        relucap=RELU_CAP,
        reluleakyness=RELU_LEAKYNESS)

print "#"*30 +" MODEL "+"#"*30
l_out, params_launits, input_layer1 = createnet()

##################################################################
#           THEANO FUNCTIONS                                     #
##################################################################
input_dict = {input_layer1: input}


cost_train = costfun(
    l_out.get_output(input_dict, deterministic=False,mask=mask),
    target_output, mask, db='COST TRAIN:')
cost_val = costfun(
    l_out.get_output(input_dict, deterministic=True,mask=mask),
    target_output, mask, db='COST VAL:')

all_params = lasagne.layers.get_all_params(l_out)

#updates = adadelta_normscaled(
#    cost_train, all_params,batch_size=BATCH_SIZE,learning_rate=1.0,
#    epsilon=10e-6, max_norm=MAX_NORM_GRADIENTS, verbose=VERBOSE)
updates = nesterov_normscaled( cost_train, all_params, 0.01, 0.5, BATCH_SIZE)
if UNITTYPE_RNN == 'LAunits':
        updates_launits = adadelta_normscaled(cost_train, params_launits,
                                batch_size=BATCH_SIZE,learning_rate=1.0, epsilon=10e-6,
                                max_norm=MAX_NORM_GRADIENTS, verbose=VERBOSE, weight_decay = LAUNITL2)
        updates.extend(updates_launits)

# print number of params
total_params = sum([p.get_value().size for p in all_params])
if UNITTYPE_RNN == 'LAunit':
    launit_params = sum([p.get_value().size for p in params_launits])
    print "#NETWORK params:", total_params, "LAunit params", launit_params
else:
    print "#NETWORK params:", total_params

logger.info('Compiling functions...')
givens = [(input, sh_input),
          (target_output, T.cast(sh_target_output, 'int32')),
         (mask, sh_mask)]
givens_preds = [(input, sh_input), (mask, sh_mask)]

train = theano.function([], cost_train, updates=updates, givens=givens,on_unused_input='warn')
compute_cost_val = theano.function([], cost_val, givens=givens, on_unused_input='warn')
compute_preds = theano.function([], l_out.get_output(input_dict, determinist=True, mask=mask),
                                  givens=givens_preds, on_unused_input='warn')

####################################################################
#           TRAINING LOOP                                          #
####################################################################
if LOAD_NETWORK !=  False:
    load_params = [all_params, params_launits]
    loadmodel(load_params,FILE_PICKLE)
logger.info('Training...')


def writecsv(postname, values):
    fn = outfolder + "/" + filename_base + "_"+postname+".csv"
    print "Writing: ", fn, "...",
    with open(fn, 'w') as csvfile:
         csvwriter = csv.writer(csvfile, delimiter=',')
         for row in values:
             csvwriter.writerow(row)
    print "OK"

confmatrix = ConfusionMatrix(N_CLASSES)
n_batches_train = int(np.ceil(float(X_train.shape[0]) / BATCH_SIZE))
n_batches_val = int(np.ceil(float(X_val.shape[0]) / BATCH_SIZE))
seq_count = 0
best_acc_val = 0
for epoch in range(N_EPOCHS):
    # single epoch training
    seq_shuffle = np.random.choice(N_SAMPLES_TRAIN_PADDED, N_SAMPLES_TRAIN_PADDED, False)
    batches_train = []
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
        if PRINT_BATCH_TIMINGS:
            batch_time =  time.time()
        c += 1
        if c % 5 == 0:
            print '%i ' % c,
        # update shared variables
        y_train_batch = np.argmax(
                y_train[ind, ], axis=-1).astype(theano.config.floatX)

        X_train_batch = X_train[ind, ]
        train_mask_batch = mask_train[ind, ]

        sh_input.set_value(X_train_batch, borrow=True)
        sh_target_output.set_value(y_train_batch, borrow=True)
        sh_mask.set_value(train_mask_batch, borrow=True)

        train()
        if PRINT_BATCH_TIMINGS:
            print time.time() - batch_time
        if VERBOSE:
            print ""
            train_preds_probs = compute_preds()
            train_mask_flat = train_mask_batch.flatten()
            train_model_preds_lab = np.argmax(train_preds_probs, axis=-1).flatten()
            train_true_preds_lab = np.argmax(y_train[ind, ], axis=-1).flatten()
            confmatrix.batchAdd(train_true_preds_lab[train_mask_flat], train_model_preds_lab[train_mask_flat])
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
        y_val_batch = np.argmax(
                 y_val[ind, ], axis=-1).astype(theano.config.floatX)


        X_val_batch = X_val[ind,]
        val_mask_batch = mask_val[ind, ]
        sh_input.set_value(X_val_batch, borrow=True)
        sh_target_output.set_value(y_val_batch, borrow=True)
        sh_mask.set_value(val_mask_batch, borrow=True)

        val_cost += compute_cost_val()
        val_preds_probs = compute_preds()
        val_model_preds_lab = np.argmax(val_preds_probs, axis=-1)
        val_true_preds_lab = np.argmax(y_val[ind, ], axis=-1)

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
    if current_acc_val > best_acc_val:
        best_acc_val = current_acc_val
        print "Saving model to: ", FILE_PICKLE, "   ",
        savemodel( [all_params, params_launits], FILE_PICKLE)
        # stack data and save it
        writecsv("val_mask", np.vstack(all_val_masks).astype('int32'))
        writecsv("best_val_model_lab", np.vstack(all_val_model_preds))
        writecsv("true_lab", np.vstack(all_val_true_preds))

    print ""
    print '------VAL ACC------->', current_acc_val
    print confmatrix
    assert np.sum(mask_val) == np.sum(confmatrix.getMat())
    confmatrix.zero()
    logger.info("Epoch {} took {}, cost = {}, acc val = {}, best acc val = {}".format(
        epoch, end_time - start_time, val_cost, current_acc_val, best_acc_val))

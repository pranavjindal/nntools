__author__ = 'soren sonderby'
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
import lasagne
from lasagne import init
import cPickle

def maskednll(output,target_output, mask,batch_size,verbose=False,db='COST:'):
    """
    Masked negative loglikelihood. Returns
    output, target_output and mask are shape (sequences x sequence_length x nclasses)
    :param output: Output from nntools network. example: last_layer,get_output(input,deterministic=False)
    :param target_output: tensor3 with one-hot-encoded targets  (sequences x sequence_length x nclasses)
    :param mask: tensor3 binary mask indicating if output should be included as error. 1 is included, 0 is excluded
                 For size of the mask is batchsize X seqlen x 1 (the last dim is broadcastable)
    :param verbose: if true prints the cross entropy
    :param db: versose printing name
    :return: symbolic expression calculating the masked negative log likelihood for a network
    """
    n_time_steps = T.sum(mask)
    total_xentropy = T.sum(T.nnet.binary_crossentropy(output, target_output) * mask)
    cross_ent =  total_xentropy / n_time_steps   #<-- FIX this
    if verbose:
        cross_ent = theano.printing.Print(db)(cross_ent)
    return cross_ent


def prederrrate(output, target_output, mask,db='Y_PRED_ERRORS:',verbose=False):
    """
    Calculates the misclassification rate. Masks 'masked' samples
    All matrices are shape (sequences x sequence_length x nclasses)
    :param output: Output from nntools network. example: last_layer,get_output(input,deterministic=False)
    :param target_output: tensor3 with one-hot-encoded targets  (sequences x sequence_length x nclasses)
    :param mask: tensor3 binary mask indicating if output should be included as error. 1 is included, 0 is excluded
    :param verbose: if true prints the cross entropy
    :param db: versose printing name
    :return:
    """
    true_labels = T.argmax(target_output, axis=-1).flatten()
    preds = T.argmax(output, axis=-1).flatten()
    eq = T.eq(true_labels,preds)
    n_time_steps = T.sum(mask)
    acc = T.sum(eq*T.squeeze(mask)) / n_time_steps

    if verbose:
        acc = theano.printing.Print(db+' ACC')(acc)
    error = 1.0-acc
    return error

# class for clipping gradient. Not uses atm
# apply around loss, T.grad(check_grad(loss), all_params)
#rom theano import ifelse
class GradClip(theano.compile.ViewOp):

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        def pgrad(g_out):
            g_out = T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound)
            g_out = ifelse(T.any(T.isnan(g_out)),
            T.ones_like(g_out)*0.00001,
            g_out)
            return g_out
        return [pgrad(g_out) for g_out in g_outs]

gradient_clipper = GradClip(-1.0, 1.0)
T.opt.register_canonicalize(theano.gof.OpRemove(gradient_clipper), name='gradient_clipper')

def momentum_normscaled(loss, all_params, lr, mom, batch_size, max_norm=np.inf, weight_decay=0.0,verbose=False):
    updates = []
    #all_grads = [theano.grad(loss, param) for param in all_params]
    all_grads = theano.grad(gradient_clipper(loss),all_params)

    grad_lst = [ T.sum( (  grad / float(batch_size) )**2  ) for grad in all_grads ]
    grad_norm = T.sqrt( T.sum( grad_lst ))
    if verbose:
        grad_norm = theano.printing.Print('MOMENTUM GRAD NORM1:')(grad_norm)

    all_grads = ifelse(T.gt(grad_norm, max_norm),
                       [grads*(max_norm / grad_norm) for grads in all_grads],
                       all_grads)


    if verbose:
        grad_lst = [ T.sum( (  grad / float(batch_size) )**2  ) for grad in all_grads ]
        grad_norm = T.sqrt( T.sum( grad_lst ))
        grad_norm = theano.printing.Print('MOMENTUM GRAD NORM2:')(grad_norm)
        all_grads = ifelse(T.gt(grad_norm, np.inf),
                           [grads*(max_norm / grad_norm) for grads in all_grads],
                           all_grads)

    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = mom * mparam_i - lr*(weight_decay*param_i + grad_i)

        updates.append( (mparam_i, v) )
        updates.append( (param_i, param_i + v) )

    return updates

def nesterov_normscaled(loss, all_params,  lr, mom, batch_size, max_norm=np.inf, weight_decay=0.0,verbose=False):
    #all_grads = [theano.grad(loss, param) for param in all_params]
    all_grads = theano.grad(gradient_clipper(loss), all_params)
    updates = []

    grad_lst = [ T.sum( (  grad / float(batch_size) )**2  ) for grad in all_grads ]
    grad_norm = T.sqrt( T.sum( grad_lst ))
    if verbose:
        grad_norm = theano.printing.Print('MOMENTUM GRAD NORM1:')(grad_norm)

    all_grads = ifelse(T.gt(grad_norm, max_norm),
                       [grads*(max_norm / grad_norm) for grads in all_grads],
                       all_grads)


    if verbose:
        grad_lst = [ T.sum( (  grad / float(batch_size) )**2  ) for grad in all_grads ]
        grad_norm = T.sqrt( T.sum( grad_lst ))
        grad_norm = theano.printing.Print('MOMENTUM GRAD NORM2:')(grad_norm)
        all_grads = ifelse(T.gt(grad_norm, np.inf),
                           [grads*(max_norm / grad_norm) for grads in all_grads],
                           all_grads)

    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        full_grad = grad_i + weight_decay * param_i
        v = mom * mparam_i - lr * full_grad # new momemtum
        w = param_i + mom * v - lr * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))

    return updates

def adadelta_normscaled(loss, all_params,batch_size=1,max_norm=np.inf,
                        max_col_norm = 0.0,learning_rate=1.0, rho=0.95,
                        epsilon=1e-6,
                        verbose=False, weight_decay=0.0):
    '''

    in the paper, no learning rate is considered (so learning_rate=1.0). Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does not become 0).

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to work for multiple datasets (MNIST, speech).

    see "Adadelta: an adaptive learning rate method" by Matthew Zeiler for more info.

    :param loss:
    :param all_params:
    :param batch_size:
    :param max_norm: Max gradient of all weights
    :param max_col_norm: contrain l2 norm of incoming weights
    :param learning_rate:
    :param rho:
    :param epsilon:
    :param verbose:
    :param weight_decay: L2 weight decay
    :return:
    '''
    def apply_max_col_norm():
        # L2 norm taken from pylearn2
        for i in range(len(all_params)):
            W = all_params[i]
            if W.ndim > 1:  # dont scale biases
                col_norms = T.sqrt(T.square(W).sum(axis=0))
                desired_norms = T.minimum(col_norms, max_col_norm)
                scale = desired_norms / T.maximum(1e-7, col_norms)
                all_params[i] *= scale

    def apply_max_norm(all_grads):
        grad_lst = [T.sum((grad/float(batch_size) )**2) for grad in all_grads]
        grad_norm = T.sqrt(T.sum( grad_lst))

        if verbose:
            grad_norm = theano.printing.Print('MOMENTUM GRAD NORM1:')(grad_norm)
        all_grads = ifelse(T.gt(grad_norm, max_norm),
                   [grads*(max_norm / grad_norm) for grads in all_grads],
                   all_grads)
        return all_grads


    #all_grads = [theano.grad(loss, param) for param in all_params]
    all_grads = theano.grad(gradient_clipper(loss),all_params)

    if max_norm > 0:
        all_grads = apply_max_norm(all_grads)

    if max_col_norm > 0:
        apply_max_col_norm()


    if verbose:
        grad_lst = [ T.sum( (  grad / float(batch_size) )**2  ) for grad in all_grads ]
        grad_norm = T.sqrt( T.sum( grad_lst ))
        grad_norm = theano.printing.Print('MOMENTUM GRAD NORM2:')(grad_norm)
        all_grads = ifelse(T.gt(grad_norm, np.inf),
                           [grads*(max_norm / grad_norm) for grads in all_grads],
                           all_grads)


    all_accumulators = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in all_params]
    all_delta_accumulators = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in all_params]

    # all_accumulators: accumulate gradient magnitudes
    # all_delta_accumulators: accumulate update magnitudes (recursive!)

    updates = []
    for param_i, grad_i, acc_i, acc_delta_i in zip(all_params, all_grads, all_accumulators, all_delta_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i**2
        updates.append((acc_i, acc_i_new))

        update_i = grad_i * T.sqrt(acc_delta_i + epsilon) / T.sqrt(acc_i_new + epsilon) # use the 'old' acc_delta here
        if weight_decay > 0:
             update_i += weight_decay * param_i

        if learning_rate > 1 or learning_rate < 1:
            updates.append((param_i, param_i - learning_rate * update_i))
        else:
            updates.append((param_i, param_i - update_i))

        acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i**2

        updates.append((acc_delta_i, acc_delta_i_new))

    return updates


def padtobatchmultiple(X, y, mask, mask_tar, batch_size):
    n_seqs = X.shape[0]
    seq_length = X.shape[1]
    n_batches_out = np.ceil(float(n_seqs) / batch_size)
    n_seqs_out = batch_size * n_batches_out

    X_out = np.zeros((n_seqs_out, seq_length, X.shape[2]), dtype=X.dtype)
    y_out = np.zeros((n_seqs_out, seq_length, y.shape[2]), dtype=y.dtype)
    mask_out = np.zeros((n_seqs_out, seq_length, 1), dtype=mask.dtype)
    mask_tar_out = np.zeros((n_seqs_out, 1, mask_tar.shape[2]), dtype=mask.dtype)

    X_out[:n_seqs, ] = X
    y_out[:n_seqs, ] = y
    mask_out[:n_seqs, ] = mask
    mask_tar_out[:n_seqs, ] = mask_tar

    return X_out, y_out, mask_out, mask_tar_out

def padtobatchmultiplesimple(X, y, mask, batch_size):
    n_seqs = X.shape[0]
    seq_length = X.shape[1]
    n_batches_out = np.ceil(float(n_seqs) / batch_size)
    n_seqs_out = batch_size * n_batches_out

    X_out = np.zeros((n_seqs_out, seq_length, X.shape[2]), dtype=X.dtype)
    y_out = np.zeros((n_seqs_out, seq_length, y.shape[2]), dtype=y.dtype)
    mask_out = np.zeros((n_seqs_out, seq_length, 1), dtype=mask.dtype)

    X_out[:n_seqs, ] = X
    y_out[:n_seqs, ] = y
    mask_out[:n_seqs, ] = mask
    return X_out, y_out, mask_out


def createmodel(rnn_layer_layers, isbrnn, batch_size,n_features,n_classes,layer_type_rnn, padded_seq_len,output_layers,learn_init,
                 input_layers,final_output_layer, input_layer_dropout=0.0,
                 dropout_rnn=0.0, output_layer_dropout=0.0, unittype_rnn='tanh',
                 relucap=0, reluleakyness=0,
                 weight_init=0.05):
    '''
    Setup various LSTM and RNN networks

    :param rnn_layer_layers: list of RNN layer sizes
    :param isbrnn: bool: if true creates bidirectional network
    :param batch_size: int, minibatch size
    :param n_features:  int, number of features
    :param n_classes: int, number of classes
    :param layer_type_rnn:  string:
        LSTMnopeeps     : LSTM without peepholes  (slower, smal mem footprint)
        LSTMnopeepsFAST : LSTM without peepholes  (faster large mem footprint)
        LSTM            : LSTM with peepholes (slower smal mem footprint)
        LSTMFAST        : LSTM with peepholes (faster large mem footprint)
        RNN                 : Vanilla recurrent network
    :param padded_seq_len: Size of padded sequences
    :param output_layers:  List of output layer sizes, added after recurrent
                           net, but before output
    :param learn_init: bool, learn initial state
    :param input_layers: List of input layer sizes, added before recurrent net
    :param final_output_layer: linear or softmax output layer
    :param input_layer_dropout: dropout on input layers
    :param dropout_rnn: dropout on RNN non recurrent connections
                        http://arxiv.org/abs/1409.2329
    :param output_layer_dropout: dropout on output layers
    :param unittype_rnn:  Unit type in recurrent nets. For LSTM this is the
        output gate.
                tanh
                sigm
                ReLU, relucap and leakyness applies
                LAunit, relucap applies, http://arxiv.org/abs/1412.6830
    :param relucap: cap on ReLU and LAunit output. For the LAunit this is the
        total output. Each line segment is capped ot be cap / segments
    :param reluleakyness: ReLU leakyness
    :param Weight init scale, float
    :return: network output layer, list of la_unit params (possbly empty list)

    '''

    l_launits = []
    def create_lstm(isbrnn, num_hidden_lst, prevlayer):

        # setup layer type
        if layer_type_rnn in ['LSTM']:
            assert False, "Not implemented"
        elif layer_type_rnn == 'LSTMFAST':
            type = 'LSTM'
            peepholes = True
        elif layer_type_rnn == 'LSTMnopeeps':
            assert False, "Not implemented"
        elif layer_type_rnn =='LSTMnopeepsFAST':
            peepholes = False
            type = 'LSTM'
        elif layer_type_rnn == 'RNN':
            type = 'RNN'
            peepholes = False
        else:
            assert False, "Unknown layertype"

        layer_func = lasagne.layers.BidirectionalLSTMLayer if isbrnn \
            else lasagne.layers.LSTMLayer
        print layer_func

        ini = lasagne.init.Uniform((-weight_init, weight_init))
        addlayer = lambda prevlayer,nhid,nonlin: layer_func(
            prevlayer, num_units=nhid, dropout_rate=dropout_rnn,
            learn_init=learn_init, nonlinearity_out=nonlin,
            peepholes = peepholes,
            W_cell_to_gates=ini, W_hid_to_gates=ini, W_in_to_gates=ini,
            b_gates=init.Constant(0.0))


        layer = prevlayer
        print "ADDING RECURRENT LAYERS WITH UNITS:", num_hidden_lst
        for lnum,nhid in enumerate(num_hidden_lst):

            # choose hidden layer unit type
            print "\t" * 2 + "Adding Recurrent layer with nhidden:", nhid
            if unittype_rnn == 'LAunit':
                print "\t" * 3 + 'Recurrent NONLIN: LAunit', relucap
                nonlin, a, b = lasagne.nonlinearities.LAunit(nhid,relucap=relucap)
                l_launits.append(a)
                l_launits.append(b)
            elif unittype_rnn == 'sigm':
                print "\t" * 3 + 'Recurrent NONLIN: sigm'
                nonlin = lasagne.nonlinearities.sigmoid
            elif unittype_rnn == 'tanh':
                print "\t" * 3 + 'Recurrent NONLIN: tanh'
                nonlin = lasagne.nonlinearities.tanh
            elif unittype_rnn == 'ReLU':
                print "\t" * 3 + 'Recurrent NONLIN: ReLU', relucap, reluleakyness
                if relucap > 0 and reluleakyness > 0:
                    print "Leaky capped"
                    nonlin = lasagne.nonlinearities.rectify_leaky_capped(relucap,reluleakyness)
                elif relucap > 0:
                    print "capped"
                    nonlin = lasagne.nonlinearities.rectify_capped(relucap)
                elif reluleakyness > 0:
                    print "Leaky"
                    nonlin = lasagne.nonlinearities.rectify_leaky(reluleakyness)
                else:
                    print "Normal ReLU"
                    nonlin = lasagne.nonlinearities.rectify
            else:
                assert False, 'Unknown Recurrent unit type: ' + unittype_rnn

            layer = addlayer(layer, nhid, nonlin)

        return layer


    def addfeedforwardlayers(input,nhid_layers,dropout, text):
        print text + " :", nhid_layers
        layer = input
        for idx,nhid in enumerate(nhid_layers):
            print "     %i) addding ffn with %i ReLU units, dropout %f" %(idx,nhid,dropout)
            layer = lasagne.layers.DenseLayer(
                layer, nhid, nonlinearity=lasagne.nonlinearities.rectify)
            if dropout > 0:
                 layer = lasagne.layers.dropout(layer, p=input_layer_dropout)
        return layer

    #add input network
    inputlayer = lasagne.layers.InputLayer(shape=(batch_size, padded_seq_len, n_features))
    input_layer_1 = inputlayer

    if type(input_layers) == list:
        #reshape to feedforward format
        inputlayer = lasagne.layers.ReshapeLayer(
            inputlayer, (batch_size * padded_seq_len, n_features))
        inputlayer = addfeedforwardlayers(
            inputlayer, input_layers, input_layer_dropout, 'FFN INPUTLAYERS:')

        inputlayer = lasagne.layers.ReshapeLayer(
            inputlayer, shape=(batch_size, padded_seq_len, inputlayer.get_output_shape()[-1]))

    lstm_out = create_lstm(isbrnn, rnn_layer_layers, inputlayer)  # forward net


    lstm_out_hid  = lstm_out.get_output_shape()[2]
    lstm_out = lasagne.layers.ReshapeLayer(lstm_out, (batch_size * padded_seq_len, lstm_out_hid))

    if type(output_layers) == list:
        lstm_out = addfeedforwardlayers(
            lstm_out, output_layers, output_layer_dropout, 'FFN OUTPUTLAYERS:')

    print "----> ADDING OUTPUT LAYER"
    if final_output_layer == 'linear':
        print "     USING LINEAR UNITS IN OUTPUT LAYER"
        nonlin_output = lasagne.nonlinearities.linear
    elif final_output_layer == 'softmax':
        print "     USING SOFTMAX UNITS IN OUTPUT LAYER"
        nonlin_output = lasagne.nonlinearities.softmax
    else:
        assert False, 'Output must be linear or softmax'

    l_output = lasagne.layers.DenseLayer(lstm_out, num_units=n_classes, nonlinearity=nonlin_output)
    net = lasagne.layers.ReshapeLayer(l_output, (batch_size, padded_seq_len, n_classes))
    return net, l_launits, input_layer_1


def savemodel(params, filename):
    print 'Saving model...',
    all_params = []
    for l in params:
        all_params.append([param.get_value() for param in l])

    with open(filename, 'wb') as f:
        cPickle.dump(all_params, f,
            protocol=cPickle.HIGHEST_PROTOCOL)
    print 'OK'

def loadmodel(old_params,fn):
    with file(fn, 'rb') as f:
        params_loaded = cPickle.load(f)
        assert len(old_params) == len(params_loaded), \
            'number of params must be the same in loaded current model'
        for op, lp,count in zip(old_params,params_loaded, range(len(old_params))):
            print "Loading param list: ", count, "with", len(lp), "params"
            assert len(lp) == len(op)
            for param, loaded_param in zip(op,lp):
                param.set_value(loaded_param)
        print "LOADING FILE", fn



def createinputwindows(x,windowwidth):
    '''
    Creates windowed data. End sequences are zero padded
    :param x: ndarray nseqs x seqlen x inputdim
    :param windowwidth:  odd int
    :return: ndarray nseqs x seqlen x (windowwindt*inputdim)
    '''
    assert windowwidth % 2 == 1, "windowwidth must be odd"
    n_seqs, max_seq_len, n_input = x.shape
    x_w = np.zeros((n_seqs,max_seq_len,windowwidth,n_input))
    for s in range(n_seqs):
        hw1 = windowwidth // 2
        for n in range(windowwidth):
                a, b = max([0, hw1]), min([0, hw1])
                x_w[s, a:max_seq_len+b, n, :] = x[s, abs(b):max_seq_len-a]
                hw1 -= 1
    x_w = x_w.reshape(( n_seqs, max_seq_len,n_input*windowwidth))
    return x_w.astype(x.dtype)




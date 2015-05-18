import pytest
from lasagne.layers import RecurrentLayer, LSTMLayer
from lasagne.layers import InputLayer
from lasagne.layers import helper
import theano
import theano.tensor as T
import numpy as np
import lasagne


def test_recurrent_return_shape():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))

    x_in = np.random.random((num_batch, seq_len, n_features)).astype('float32')
    mask_in = np.random.random((num_batch, seq_len)).astype('float32')

    l_rec = RecurrentLayer(l_inp, num_units=num_units)
    l_out = helper.get_output(l_rec, x, mask=mask)
    f_rec = theano.function([x, mask], l_out)
    f_out = f_rec(x_in, mask_in)

    assert f_out.shape == (num_batch, seq_len, num_units)


def test_recurrent_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_rec = RecurrentLayer(l_inp,
                           num_units=num_units)
    l_out = helper.get_output(l_rec, x, mask=mask)
    g = T.grad(T.mean(l_out), lasagne.layers.get_all_params(l_rec))
    assert isinstance(g, (list, tuple))


def test_recurrent_nparams():
    l_inp = InputLayer((1, 2, 3))
    l_rec = RecurrentLayer(l_inp, 5, learn_init=False)
    assert len(lasagne.layers.get_all_params(l_rec, trainable=True)) == 3
    assert len(lasagne.layers.get_all_params(l_rec, regularizable=False)) == 1


def test_recurrent_nparams_learn_init():
    l_inp = InputLayer((1, 2, 3))
    l_rec = RecurrentLayer(l_inp, 5, learn_init=True)
    assert len(lasagne.layers.get_all_params(l_rec, trainable=True)) == 4
    assert len(lasagne.layers.get_all_params(l_rec, regularizable=False)) == 2


def test_lstm_return_shape():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))

    x_in = np.random.random((num_batch, seq_len, n_features)).astype('float32')
    mask_in = np.random.random((num_batch, seq_len)).astype('float32')

    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    l_out = helper.get_output(l_lstm, x, mask=mask)
    f_lstm = theano.function([x, mask], l_out)
    f_out = f_lstm(x_in, mask_in)

    assert f_out.shape == (num_batch, seq_len, num_units)


def test_lstm_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    l_out = helper.get_output(l_lstm, x, mask=mask)
    g = T.grad(T.mean(l_out), lasagne.layers.get_all_params(l_lstm))
    assert isinstance(g, (list, tuple))


def test_lstm_nparams_no_peepholes():
    l_inp = InputLayer((1, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=False, learn_init=False)
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 12
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_nparams_peepholes():
    l_inp = InputLayer((1, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=True, learn_init=False)
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 15
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_nparams_learn_init():
    l_inp = InputLayer((1, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=False, learn_init=True)
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 14
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6

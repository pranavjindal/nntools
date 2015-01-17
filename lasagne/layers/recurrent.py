import theano
import theano.tensor as T
import numpy as np
from .. import nonlinearities
from .. import init

from .base import Layer
from . import helper

from lasagne import utils
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng = RandomStreams()


class RecurrentLayer(Layer):
    '''
    A layer which implements a recurrent connection.

    Expects inputs of shape
        (n_batch, n_time_steps, n_features_1, n_features_2, ...)
    '''
    def __init__(self, input_layer, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False):
        '''
        Create a recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the recurrent layer
            - input_to_hidden : nntools.layers.Layer
                Layer which connects input to thie hidden state
            - hidden_to_hidden : nntools.layers.Layer
                Layer which connects the previous hidden state to the new state
            - nonlinearity : function or theano.tensor.elemwise.Elemwise
                Nonlinearity to apply when computing new state
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : boolean
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
        '''
        super(RecurrentLayer, self).__init__(input_layer)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        (n_batch, self.num_units) = self.input_to_hidden.get_output_shape()

        # Initialize hidden state
        self.hid_init = self.create_param(hid_init, (n_batch, self.num_units))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = (helper.get_all_params(self.input_to_hidden) +
                helper.get_all_params(self.hidden_to_hidden))

        if self.learn_init:
            return params + self.get_init_params()
        else:
            return params

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return (helper.get_all_bias_params(self.input_to_hidden) +
                helper.get_all_bias_params(self.hidden_to_hidden))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, mask=None, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  This is
                needed when scanning backwards.  If all sequences are of the
                same length, it should be all 1s.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        if self.backwards:
            assert mask is not None, ("Mask must be given to get_output_for"
                                      " when backwards is true")

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        if self.backwards:
            mask = mask.dimshuffle(1, 0, 'x')

        # Create single recurrent computation step function
        def step(layer_input, hid_previous):
            return self.nonlinearity(
                self.input_to_hidden.get_output(layer_input) +
                self.hidden_to_hidden.get_output(hid_previous))

        def step_back(layer_input, mask, hid_previous):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = (step(layer_input, hid_previous)*mask
                   + hid_previous*(1 - mask))
            return [hid]

        if self.backwards:
            sequences = [input, mask]
            step_fun = step_back
        else:
            sequences = input
            step_fun = step


        output = theano.scan(step_fun, sequences=sequences,
                             go_backwards=self.backwards,
                             outputs_info=[self.hid_init])[0]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        if self.backwards:
            output = output[:, ::-1, :]

        return output


class ReshapeLayer(Layer):
    def __init__(self, input_layer, shape):
        super(ReshapeLayer, self).__init__(input_layer)
        self.shape = shape

    def get_output_shape_for(self, input_shape):
        return self.shape

    def get_output_for(self, input, *args, **kwargs):
        return input.reshape(self.shape)


class LSTMLayer(Layer):
    '''
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [#graves2014generating]_, which is
    the current common definition. Gate names are taken from [#zaremba2014],
    figure 1. This is the unidirectional layer, for bidirectional
    implementation see BidirectionalLSTMLayer.

    :references:
        .. [#graves2014generating] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks".
        .. [#zareba2014] Zaremba, W. et.al  Recurrent neural network
           regularization. (http://arxiv.org/abs/1409.2329)
    '''
    ini = init.Uniform((-0.05, 0.05))
    zero = init.Constant(0.)
    ortho = init.Orthogonal(np.sqrt(2))
    def __init__(self, input_layer, num_units,
                 W_in_to_gates=ini,
                 W_hid_to_gates=ini,
                 W_cell_to_gates=ini,
                 b_gates=zero,
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 nonlinearity_modulationgate=nonlinearities.tanh,
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 learn_init=True,
                 peepholes=True,
                 dropout_rate=0.0,
                 dropout_rescale=True):
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_in_to_gates : function or np.ndarray or theano.shared
            - W_hid_to_gates : function or np.ndarray or theano.shared
            - W_cell_to_gates : function or np.ndarray or theano.shared
            - b_gates : function or np.ndarray or theano.shared
            - nonlinearity_ingate : function
            - nonlinearity_forgetgate : function
            - nonlinearity_modulationgate : function
            - nonlinearity_outgate : function
            - nonlinearity_out : function
            - cell_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - hid_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - learn_init : boolean
                If True, initial hidden values are learned
            - peepholes : boolean
                If True, the LSTM uses peephole connections.
                When False, W_cell_to_ingate, W_cell_to_forgetgate and
                W_cell_to_outgate are ignored.
            - dropout_rate: dropout rate on non-recurrent connectios
            - dropout_rescale: if dropout activation should be rescaled or not
        '''

        # Initialize parent layer
        super(LSTMLayer, self).__init__(input_layer)
        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_modulationgate is None:
            self.nonlinearity_modulationgate = nonlinearities.identity
        else:
            self.nonlinearity_modulationgate = nonlinearity_modulationgate

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate

        if nonlinearity_out is None:
            self.nonlinearity_out = nonlinearities.identity
        else:
            self.nonlinearity_out = nonlinearity_out

        self.learn_init = learn_init
        self.num_units = num_units
        self.peepholes = peepholes
        self.dropout_rate = dropout_rate
        self.dropout_rescale = dropout_rescale

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        if self.peepholes:
            self.W_cell_to_gates = self.create_param(
                W_cell_to_gates, (3*num_units))
        self.b_gates = self.create_param(b_gates, (4*num_units))
        self.W_hid_to_gates = self.create_param(
            W_hid_to_gates, (num_units, 4*num_units))
        self.W_in_to_gates = self.create_param(
            W_in_to_gates, (num_inputs,4*num_units))

        # Setup initial values for the cell and the lstm hidden units
        self.cell_init = self.create_param(cell_init, (num_batch, num_units))
        self.hid_init = self.create_param(hid_init, (num_batch, num_units))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()
        if self.peepholes:
            params.extend(self.get_peephole_params())

        if self.learn_init:
            params.extend(self.get_init_params())

        return params

    def get_weight_params(self):
        '''
        Get all weights of this layer
        :returns:
            - weight_params : list of theano.shared
                List of all weight parameters
        '''
        return [self.W_in_to_gates, self.W_hid_to_gates]

    def get_peephole_params(self):
        '''
        Get all peephole parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all peephole parameters
        '''
        return [self.W_cell_to_gates]

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init, self.cell_init]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b_gates]

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # dropout function
        def D(x):
            batch_size, seq_length, _ = self.input_layer.get_output_shape()
            shape = (seq_length, batch_size, 4*self.num_units)
            retain_prob = 1 - self.dropout_rate
            if self.dropout_rescale:
                x /= retain_prob
            return x * utils.floatX(
                _srng.binomial(shape, p=retain_prob, dtype='int32'))

        # precompute inputs*W and dimshuffle
        # Input is provided as (n_batch, n_time_steps, n_features)
        # W _in_to_gates is (n_features, 4*num_units). input dot W is then
        # (n_batch, n_time_steps, 4*num_units). Because scan iterate over the
        # first dimension we dimshuffle to (n_time_steps, n_batch, n_features)
        if deterministic or self.dropout_rate == 0:
            input_dot_W = T.dot(input, self.W_in_to_gates).dimshuffle(1, 0, 2)
        else:
            input_dot_W = D(
                T.dot(input, self.W_in_to_gates).dimshuffle(1, 0, 2))


        input_dot_W += self.b_gates


        # input_dow_w is (n_batch, n_time_steps, 4*num_units). We define a
        # slicing function that extract the input to each LSTM gate
        # slice_c is similar but for peephole weights.
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def slice_c(x, n):
            return x[n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_dot_W_n is the n'th row of the input dot W multiplication
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        #
        # Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
        def step(input_dot_W_n, cell_previous, hid_previous):

            # calculate gates pre-activations and slice
            gates = input_dot_W_n + T.dot(hid_previous, self.W_hid_to_gates)
            ingate = slice_w(gates,0)
            forgetgate = slice_w(gates,1)
            modulationgate = slice_w(gates,2)
            outgate = slice_w(gates,3)


            if self.peepholes:
                ingate += cell_previous*slice_c(self.W_cell_to_gates, 0)
                forgetgate = cell_previous*slice_c(self.W_cell_to_gates, 1)
                outgate = cell_previous*slice_c(self.W_cell_to_gates, 2)

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            modulationgate = self.nonlinearity_modulationgate(modulationgate)
            outgate = self.nonlinearity_outgate(outgate)

            cell = forgetgate*cell_previous + ingate*modulationgate
            hid = outgate*self.nonlinearity_out(cell)
            return [cell, hid]

        # Scan op iterates over first dimension of input and repeatedly
        # applied the step function
        output = theano.scan(step, sequences=input_dot_W,
                             outputs_info=[self.cell_init, self.hid_init],
                             go_backwards=False)[0][1]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        return output


class BidirectionalLSTMLayer(Layer):
    '''
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [#graves2014generating]_, which is
    the current common definition. Gate names are taken from [#zaremba2014],
    figure 1. This is the bidirectional layer, for unidirectional
    implementation see LSTMLayer.
    The output from the forward and backward passes are concatenated, such
    that the output will be 2*num_units in the last dimension of the output.

    :references:
        .. [#graves2014generating] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks".
        .. [#zareba2014] Zaremba, W. et.al  Recurrent neural network
           regularization. (http://arxiv.org/abs/1409.2329)
    '''
    ini = init.Uniform((-0.05, 0.05))
    zero = init.Constant(0.)
    ortho = init.Orthogonal(np.sqrt(2))
    def __init__(self, input_layer, num_units,
                 W_in_to_gates=ini,
                 W_hid_to_gates=ini,
                 W_cell_to_gates=ini,
                 b_gates=zero,
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 nonlinearity_modulationgate=nonlinearities.tanh,
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init_fwd=init.Constant(0.),
                 hid_init_fwd=init.Constant(0.),
                 cell_init_bck=init.Constant(0.),
                 hid_init_bck=init.Constant(0.),
                 learn_init=True,
                 peepholes=True,
                 dropout_rate=0.0,
                 dropout_rescale=True):
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_in_to_gates : function or np.ndarray or theano.shared
            - W_hid_to_gates : function or np.ndarray or theano.shared
            - W_cell_to_gates : function or np.ndarray or theano.shared
            - b_gates : function or np.ndarray or theano.shared
            - nonlinearity_ingate : function
            - nonlinearity_forgetgate : function
            - nonlinearity_modulationgate : function
            - nonlinearity_outgate : function
            - nonlinearity_out : function
            - init : function or np.ndarray or theano.shared
            - hid_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - learn_init : boolean
                If True, initial hidden values are learned
            - peepholes : boolean
                If True, the LSTM uses peephole connections.
                When False, W_cell_to_ingate, W_cell_to_forgetgate and
                W_cell_to_outgate are ignored.
            - dropout_rate: dropout probability on non recurrent conncections
            - dropout_rescale: rescale dropout connections or not
        '''


        # Initialize parent layer
        super(BidirectionalLSTMLayer, self).__init__(input_layer)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_modulationgate is None:
            self.nonlinearity_modulationgate = nonlinearities.identity
        else:
            self.nonlinearity_modulationgate = nonlinearity_modulationgate

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate

        if nonlinearity_out is None:
            self.nonlinearity_out = nonlinearities.identity
        else:
            self.nonlinearity_out = nonlinearity_out

        self.learn_init = learn_init
        self.num_units = num_units
        self.peepholes = peepholes
        self.dropout_rate = dropout_rate
        self.dropout_rescale = dropout_rescale

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        if self.peepholes:
            self.W_cell_to_gates = self.create_param(
                W_cell_to_gates, (2, 3*num_units))
        self.b_gates = self.create_param(
            b_gates, (2, 4*num_units))
        self.W_hid_to_gates = self.create_param(
            W_hid_to_gates, (2, num_units, 4*num_units))
        self.W_in_to_gates = self.create_param(
            W_in_to_gates, (2, num_inputs, 4*num_units))

        # Setup initial values for the cell and the lstm hidden units
        self.cell_init_fwd = self.create_param(
            cell_init_fwd, (num_batch, num_units))
        self.hid_init_fwd = self.create_param(hid_init_fwd, (num_batch, num_units))
        self.cell_init_bck = self.create_param(
            cell_init_bck, (num_batch, num_units))
        self.hid_init_bck = self.create_param(hid_init_bck, (num_batch, num_units))
        #names for debugging
        if self.peepholes:
            self.W_cell_to_gates.name = "W_cell_to_gates"
        self.b_gates.name = "b_gates"
        self.W_hid_to_gates.name = "W_hid_to_gates"
        self.W_in_to_gates.name = "W_in_to_gates"

        self.cell_init_fwd.name = "cell_init_fwd"
        self.hid_init_fwd.name = "hid_init_fwd"
        self.cell_init_bck.name = "cell_init_bck"
        self.hid_init_bck.name = "hid_init_bck"

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()
        if self.peepholes:
            params.extend(self.get_peephole_params())

        if self.learn_init:
            params.extend(self.get_init_params())

        return params

    def get_weight_params(self):
        '''
        Get all weights of this layer
        :returns:
            - weight_params : list of theano.shared
                List of all weight parameters
        '''
        return [self.W_in_to_gates, self.W_hid_to_gates]

    def get_peephole_params(self):
        '''
        Get all peephole parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all peephole parameters
        '''
        return [self.W_cell_to_gates, self.W_cell_to_gates]

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init_fwd, self.cell_init_fwd,
                self.hid_init_bck, self.cell_init_bck]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b_gates]

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return (input_shape[0], input_shape[1], 2*self.num_units)

    def get_output_for(self, input_fwd, mask=None, deterministic=False,
                       *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  This is
                needed when scanning backwards.  If all sequences are of the
                same length, it should be all 1s.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''

        assert mask is not None, "Mask must be given for bidirectional layer"

        # Treat all layers after the first as flattened feature dimensions
        if input_fwd.ndim > 3:
            input_fwd = input.reshape((input_fwd.shape[0], input_fwd.shape[1],
                                   T.prod(input_fwd.shape[2:])))

        # dropout function
        def D(x):
            batch_size, seq_length, _ = self.input_layer.get_output_shape()
            shape = (seq_length, batch_size, 4*self.num_units)
            retain_prob = 1 - self.dropout_rate
            if self.dropout_rescale:
                x /= retain_prob
            return x * utils.floatX(
                _srng.binomial(shape, p=retain_prob, dtype='int32'))

        # precompute inputs*W and dimshuffle
        # Input is provided as (n_batch, n_time_steps, n_features)
        # W _in_to_gates is (n_features, 4*num_units). input dot W is then
        # (n_batch, n_time_steps, 4*num_units). Because scan iterate over the
        # first dimension we dimshuffle to (n_time_steps, n_batch, n_features)
        # flip input and mask if we ar going backwards
        input_bck = input_fwd[:, ::-1, :]
        mask_bck = mask[:, ::-1]
        #set_subtensor(x, y, inplace=False, tolerate_inplace_aliasing=False)
        if deterministic or self.dropout_rate == 0:
            input_dot_W_fwd = T.dot(
                input_fwd, self.W_in_to_gates[0]).dimshuffle(1, 0, 2)
            input_dot_W_bck = T.dot(
                input_bck, self.W_in_to_gates[1]).dimshuffle(1, 0, 2)
        else:
            input_dot_W_fwd = D(
                T.dot(input_fwd, self.W_in_to_gates[0]).dimshuffle(1, 0, 2))
            input_dot_W_bck = D(
                T.dot(input_bck, self.W_in_to_gates[1]).dimshuffle(1, 0, 2))

        input_dot_W_fwd += self.b_gates[0]
        input_dot_W_bck += self.b_gates[1]


        # mask is given as (batch_size, seq_len) or (batch_size, seq_len).
        # Because scan iterates over
        # first dim. If mask is 2d we dimshuffle to (seq_len, batch_size) and
        # add a broadcastable dimension. If 3d assume that third dim is
        # broadcastable.
        if mask_bck.ndim == 2:
            mask_bck = mask_bck.dimshuffle(1, 0, 'x')
        else:
            assert mask_bck.broadcastable == (False, False, True), \
                "When mask is 3d the last dimension must be boadcastable"
            mask_bck = mask_bck.dimshuffle(1, 0, 2)
            mask_bck.name = 'mask_bck'


        # input_dow_w is (n_batch, n_time_steps, 4*num_units). We define a
        # slicing function that extract the input to each LSTM gate
        # slice_c is similar but for peephole weights.
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def slice_c(x, n):
            return x[n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function.
        # Calculates both the forward and the backward pass.
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        #
        # Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
        def dostep(input_dot_W_n, cell_previous, hid_previous, idx):

            # calculate gates pre-activations and slice
            gates = input_dot_W_n + T.dot(hid_previous, self.W_hid_to_gates[idx])
            ingate = slice_w(gates,0)
            forgetgate = slice_w(gates,1)
            modulationgate = slice_w(gates,2)
            outgate = slice_w(gates,3)

            if self.peepholes:
                ingate += cell_previous*slice_c(self.W_cell_to_gates[idx], 0)
                forgetgate = cell_previous*slice_c(self.W_cell_to_gates[idx], 1)
                outgate = cell_previous*slice_c(self.W_cell_to_gates[idx], 2)

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            modulationgate = self.nonlinearity_modulationgate(modulationgate)
            outgate = self.nonlinearity_outgate(outgate)

            cell = forgetgate*cell_previous + ingate*modulationgate
            hid = outgate*self.nonlinearity_out(cell)
            return cell, hid


        def step(input_dot_W_fwd, input_dot_W_bck, mask_bck,
                cell_previous_fwd, hid_previous_fwd,
                cell_previous_bck, hid_previous_bck):

            #forward
            cell_fwd, hid_fwd = dostep(
                input_dot_W_fwd, cell_previous_fwd, hid_previous_fwd, 0)
            # backward
            cell_bck, hid_bck = dostep(
                input_dot_W_bck, cell_previous_bck, hid_previous_bck, 1)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask_bck = 1 - mask_bck
            cell_bck = cell_bck*mask_bck + cell_previous_bck*not_mask_bck
            hid_bck = hid_bck*mask_bck + hid_previous_bck*not_mask_bck

            return [cell_fwd, cell_bck, hid_bck, hid_fwd]

        sequences = [input_dot_W_fwd, input_dot_W_bck, mask_bck]
        init = [self.cell_init_fwd, self.cell_init_bck,
                self.hid_init_fwd, self.hid_init_bck]

        # Scan op iterates over first dimension of input and repeatedly
        # applied the step function
        scan_out = theano.scan(step, sequences=sequences, outputs_info=init)

        # output is  (n_time_steps, n_batch, n_units))
        output_fwd = scan_out[0][2]
        output_bck = scan_out[0][3]

        # reverse bck output
        output_bck = output_bck[::-1, :, :]

        # concateante fwd and bck
        output = T.concatenate([output_fwd, output_bck], axis=2)


        # Now, dimshuffle back to (n_batch, n_time_steps, n_units))
        output = output.dimshuffle(1, 0, 2)

        return output
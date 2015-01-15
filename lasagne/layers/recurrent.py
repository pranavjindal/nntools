import theano
import theano.tensor as T

from .. import nonlinearities
from .. import init
import numpy as np
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


class RECURRENTTEST(Layer):
    """
    FAST is 30% percent faster + alleviates some gpu dropout problems but uses
    more memory. Implementation based on work by Colin Raffel.


    :references:
        .. [#graves2014generating] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks".
        .. [Zaremba2014] Zaremba, W., Sutskever, I., & Vinyals, O. (2014).
            Recurrent neural network regularization.
    """
    ini = init.Uniform((-0.05, 0.05))
    glorot = init.Uniform()
    zero = init.Constant(0.)
    ortho = init.Orthogonal(np.sqrt(2))
    # ini = init.Normal(0.1)
    sigm = nonlinearities.sigmoid
    tanh = nonlinearities.tanh

    def __init__(self, input_layer, num_units, backwards=False, W_x=ini,
                 non_lin_i=sigm, non_lin_f=sigm, non_lin_g=tanh,W_h=ini,
                 W_c=ini, b=zero, non_lin_o=sigm, non_lin_out=tanh,
                 c_init=zero, h_init=zero, dropout_rate=0,
                 dropout_rescale=True, learn_init=True, peepholes=False,
                 type='LSTM'):
        """
        Initialize an LSTM layer optionally with peepholes or RNN.
        For details on LSTM see [#Zaremba2014] sec. 3.1 + figure 1
        and [#graves2014generator] eq 7-11.

        Optionally applies dropout to the non recurrent connections

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - backwards : Boolean
                indicate if scan should be forwards or backwards, for bidirectional LSTM
            - W_x : function or np.ndarray or theano.shared
                :math:`W_{x}`
            - W_h : function or np.ndarray or theano.shared
                :math:`W_{h}`
            - b : function or np.ndarray or theano.shared
                :math:`b`
            - non_lin_i : function
                :math:`\sigma`
            - non_lin_f : function
                :math:`\sigma`
            - non_lin_g : function
            - non_lin_o : function
                :math:`\sigma`
            - non_lin_out : function or np.ndarray or theano.shared
                :math:`\tanh`
            - c_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - h_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - dropout_rate : float between 0 and 1
                Determines the rate of dropout. 1 will dropout all units, 0 will dropout none [#Zaremba2014]
            - dropout_rescale: Boolean
                if true the non recurrent weights are scaled with 1/(1 - dropout_rate) when deterministic is False
            - learn_init: Boolean
                if true the initial state for hidden and memory cells will be learned
            - peepholes: Boolean
                Using peepholes seems to increase the computational time by ~25%

        """

        super(RECURRENTTEST, self).__init__(input_layer)
        assert type in ['LSTM', 'RNN'], 'TYPE must be LSTM or RNN'
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.dropout_rescale = dropout_rescale
        self.backwards = backwards
        self.learn_init = learn_init
        self.type = type
        self.peepholes = peepholes
        print '       |TYPE       ',  self.type
        print '       |DROPOUT    ',  self.dropout_rate
        print '       |BACKWARDS  ',  self.backwards
        print '       |LEARN INIT ',  self.learn_init
        print '       |NUNITS     ',  self.num_units
        if self.type == 'LSTM':
            print '       |PEEPHOLES  ',  self.peepholes

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        # Initialize parameters using the supplied args
        self.h_init = self.create_param(h_init, (num_batch, num_units))
        if self.type == 'LSTM':
            self.c_init = self.create_param(c_init, (num_batch, num_units))
            self.init_params = [self.c_init, self.h_init]
            self.n_gates = 4
            self.non_lin_i = non_lin_i
            self.non_lin_f = non_lin_f
            self.non_lin_o = non_lin_g
            self.non_lin_g = non_lin_o
            if self.peepholes:
                self.W_c = self.create_param(W_c, ((self.n_gates-1)*num_units))


        elif self.type == 'RNN':
            assert peepholes== False, "RNN's cannot have peepholes"
            self.n_gates = 1
            self.init_params = [self.h_init]

        self.b = self.create_param(b, (self.n_gates*num_units))
        self.W_h = self.create_param(W_h, (num_units, self.n_gates*num_units))
        self.W_x = self.create_param(W_x, (num_inputs,self.n_gates*num_units))
        self.non_lin_out = non_lin_out

    def get_weight_params(self):
        return [self.W_h, self.W_x]

    def get_init_params(self):
        return self.init_params


    def get_bias_params(self):
        return [self.b]

    def get_peephole_params(self):
        return [self.W_c]

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()

        if self.learn_init:
            params += self.get_init_params()

        if self.peepholes:
            params += self.get_peephole_params()

        return params

    def get_output_shape_for(self, input_shape):
        """
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        """
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, deterministic=False,mask=None, *args, **kwargs):
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
            - deterministic : boolean
                Controls if dropout if applied or not, Should be true when
                testing.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''

        if self.backwards and mask == None:
            assert False

        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Create single recurrent computation step function
        if self.type == 'LSTM':
            def sliceW(x, n):
                return x[:, n*self.num_units:(n+1)*self.num_units]
            def slicePeep(x, n):
                return x[n*self.num_units:(n+1)*self.num_units]
            def step(W_xx_n, c_pt, h_pt,):
                # size
                #   W_xx_n: batch_size, 3*self.num_units
                #   c_pt, h_pt  batch_size x n_hid
                gates = W_xx_n + T.dot(h_pt, self.W_h)
                i_t = self.non_lin_i(sliceW(gates, 0))
                f_t = self.non_lin_f(sliceW(gates, 1))
                o_t = self.non_lin_o(sliceW(gates, 2))
                g_t = self.non_lin_g(sliceW(gates, 3))
                if self.peepholes:
                    i_t += c_pt*slicePeep(self.W_c, 0)
                    f_t += c_pt*slicePeep(self.W_c, 1)
                    o_t += c_pt*slicePeep(self.W_c, 2)
                c_t = (f_t*c_pt + i_t*g_t)
                h_t = o_t*self.non_lin_out(c_t)
                return [c_t, h_t]
        def stepbck(W_xx_n, mask, c_pt, h_pt):
            c_t, h_t = step(W_xx_n, c_pt, h_pt)
            not_mask = 1-mask
            c_out = c_t*mask + not_mask*c_pt
            h_out = h_t*mask + not_mask*h_pt
            return [c_out, h_out]


        if self.type == 'RNN':
            def step(W_xx_n, h_pt):
                h_t = self.non_lin_out(W_xx_n + T.dot(h_pt, self.W_h))
                return [h_t]
            def stepbck(W_x, mask, h_pt):
                h_t  = step(W_x, h_pt)[0]
                not_mask = 1-mask
                h_out = h_t*mask + not_mask*h_pt
                return [h_out]


        def D(x):
            batch_size, seq_length, num_inputs = self.input_layer.get_output_shape()
            shape = (seq_length, batch_size, self.n_gates*self.num_units)
            retain_prob = 1 - self.dropout_rate
            if self.dropout_rescale:
                x /= retain_prob
            return x * utils.floatX(
                _srng.binomial(shape, p=retain_prob, dtype='int32'))


        # input is batch_size batch_size x seq_len x n_input
        if deterministic or self.dropout_rate == 0:
            W_xx = T.dot(input,self.W_x).dimshuffle(1, 0, 2) + self.b
        else:
            W_xx = D(T.dot(input,self.W_x).dimshuffle(1, 0, 2)) + self.b

        if self.backwards:
            mask = mask.dimshuffle(1, 0, 2)
            sequences = [W_xx, mask]
            fun = stepbck
        else:  # fwd
            sequences = [W_xx]
            fun = step

        output = theano.scan(
            fun, sequences=sequences, go_backwards=self.backwards,
            outputs_info=self.init_params)

        if self.type == 'LSTM':
            output = output[0][1]
        elif self.type == 'RNN':
            output = output[0]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        # always return output from t=0 to t=n
        if self.backwards:
            output = output[:, ::-1, :]
        return output


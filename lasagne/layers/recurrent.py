"""
Layers to construct recurrent networks. Recurrent layers can be used similarly
to feed-forward layers except that the input shape is expected to be
`(batch_size, sequence_length, num_inputs). The input is allowed to have more
than three dimensions in which case dimensions trailing the third dimension are
flattened.

The following recurrent layers are implemented:

* :func:`CustomRecurrentLayer()`
* :func:`RecurrentLayer()`
* :func:`LSTMLayer()`

Recurrent layers and feed-forward layers can be combined in the same network
by using a few reshape operations, please refer to the recurrent examples for
further explanations.

"""
import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init

from .base import Layer
from .input import InputLayer
from .dense import DenseLayer
from . import helper

__all__ = [
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "LSTMLayer"
]


class CustomRecurrentLayer(Layer):
    """
    A layer which implements a recurrent connection.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    input_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects input to the hidden state
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
    nonlinearity : function or theano.tensor.elemwise.Elemwise
        Nonlinearity to apply when computing new state
    hid_init : function or np.ndarray or theano.shared
        Initial hidden state
    backwards : boolean
        If True, process the sequence backwards
    learn_init : boolean
        If True, initial hidden values are learned
    gradient_steps : int
        Number of timesteps to include in backpropagated gradient
        If -1, backpropagate through the entire sequence
    return_sequence : boolean
        Specifies if the recurrent layer should output the hidden state for all
        positions in the sequence or only for the last position. If true the
        output shape is (num_batch, sequence_lengt, num_units) if false the
        output shape is (num_batch, num_units).
    grad_clipping: False or float
        If float the gradient messages are clipped during the backward pass.
        See [1]_ (p. 6) for further explanation.

    References
    ----------
    .. [1] Alex Graves : Generating Sequences With Recurrent Neural
           Networks
    """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 return_sequence=True,
                 grad_clipping=False):

        super(CustomRecurrentLayer, self).__init__(incoming)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.return_sequence = return_sequence
        self.grad_clipping = grad_clipping

        # check that output shapes match
        i2h_out = helper.get_output_shape(input_to_hidden)
        h2h_out = helper.get_output_shape(hidden_to_hidden)
        if i2h_out != h2h_out:
            raise ValueError("The output shape for input_to_hidden and "
                             "input_to_hidden must be equal was, ",
                             i2h_out, h2h_out)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        (self.n_batch, _, self.num_inputs) = self.input_shape
        self.num_units = i2h_out[-1]

        # Initialize hidden state
        self.hid_init = self.add_param(hid_init, (1, self.num_units),
                                       trainable=learn_init, name="hid_init")

    def get_params(self, **tags):
        params = (helper.get_all_params(self.input_to_hidden, **tags) +
                  helper.get_all_params(self.hidden_to_hidden, **tags))

        if self.learn_init:
            return params + [self.hid_init]
        else:
            return params

    def get_output_shape_for(self, input_shape):
        if self.return_sequence:
            return input_shape[0], input_shape[1], self.num_units
        else:
            return input_shape[0], self.num_units

    def get_output_for(self, input, mask=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If None,
            then it assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape (n_batch, n_time_steps) where
            `mask[i, j] = 1` when `j <= (length of sequence i)` and
            `mask[i, j] = 0` when `j > (length of sequence i)`.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable
        """
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Create single recurrent computation step function
        def step(layer_input, hid_previous):
            i2h = helper.get_output(
                self.input_to_hidden, layer_input, **kwargs)
            h2h = helper.get_output(
                self.hidden_to_hidden, hid_previous, **kwargs)
            hid_pre = i2h + h2h

            # clip gradients
            if self.grad_clipping is not False:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)
            return self.nonlinearity(hid_pre)

        def step_masked(layer_input, mask_n, hid_previous):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = (step(layer_input, hid_previous)*mask_n +
                   hid_previous*(1 - mask_n))
            return [hid]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        # repeat hid_init to batch size
        hid_init = T.dot(T.ones((self.n_batch, 1)), self.hid_init)

        hid_out = theano.scan(
            step_fun,
            sequences=sequences,
            go_backwards=self.backwards,
            outputs_info=[hid_init],
            truncate_gradient=self.gradient_steps)[0]

        if self.return_sequence:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1, :]
        else:
            # no need to dimshuffle because we return the last state
            hid_out = hid_out[-1]
        return hid_out


class RecurrentLayer(CustomRecurrentLayer):
    """
    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        Number of hidden units in the layer
    W_in_to_hid : function or np.ndarray or theano.shared
        Initializer for input-to-hidden weight matrix
    W_hid_to_hid : function or np.ndarray or theano.shared
        Initializer for hidden-to-hidden weight matrix
    b : function or np.ndarray or theano.shared
        Initializer for bias vector
    nonlinearity : function or theano.tensor.elemwise.Elemwise
        Nonlinearity to apply when computing new state
    hid_init : function or np.ndarray or theano.shared
        Initial hidden state
    backwards : boolean
        If True, process the sequence backwards
    learn_init : boolean
        If True, initial hidden values are learned
    gradient_steps : int
        Number of timesteps to include in backpropagated gradient
        If -1, backpropagate through the entire sequence
    return_sequence : boolean
        Specifies if the recurrent layer should output the hidden state for all
        positions in the sequence or only for the last position. If true the
        output shape is (num_batch, sequence_lengt, num_units) if false the
        output shape is (num_batch, num_units).
    grad_clipping: False or float
        If float the gradient messages are clipped during the backward pass.
        See [1]_ (p. 6) for further explanation.

    References
    ----------
    .. [1] Alex Graves : Generating Sequences With Recurrent Neural
           Networks
    """
    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 return_sequence=True,
                 grad_clipping=False):
        input_shape = helper.get_output_shape(incoming)
        n_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((n_batch,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((n_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None)

        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps, return_sequence=return_sequence,
            grad_clipping=grad_clipping)


class LSTMLayer(Layer):
    """
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [1]_, which is
    the current common definition. Gate names are taken from [2],
    figure 1.

    Parameters
    ----------
    incoming : a :class:`:class:`lasagne.layers.Layer`` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        Number of hidden units in the layer
    W_in_to_ingate : function or np.ndarray or theano.shared
        :math:`W_{xi}`
    W_hid_to_ingate : function or np.ndarray or theano.shared
        :math:`W_{hi}`
    W_cell_to_ingate : function or np.ndarray or theano.shared
        :math:`W_{ci}`
    b_ingate : function or np.ndarray or theano.shared
        :math:`b_i`
    nonlinearity_ingate : function
        :math:`\sigma`
    W_in_to_forgetgate : function or np.ndarray or theano.shared
        :math:`W_{xf}`
    W_hid_to_forgetgate : function or np.ndarray or theano.shared
        :math:`W_{hf}`
    W_cell_to_forgetgate : function or np.ndarray or theano.shared
        :math:`W_{cf}`
    b_forgetgate : function or np.ndarray or theano.shared
        :math:`b_f`
    nonlinearity_forgetgate : function
        :math:`\sigma`
    W_in_to_cell : function or np.ndarray or theano.shared
        :math:`W_{ic}`
    W_hid_to_cell : function or np.ndarray or theano.shared
        :math:`W_{hc}`
    b_cell : function or np.ndarray or theano.shared
        :math:`b_c`
    nonlinearity_cell : function or np.ndarray or theano.shared
        :math:`\tanh`
    W_in_to_outgate : function or np.ndarray or theano.shared
        :math:`W_{io}`
    W_hid_to_outgate : function or np.ndarray or theano.shared
        :math:`W_{ho}`
    W_cell_to_outgate : function or np.ndarray or theano.shared
        :math:`W_{co}`
    b_outgate : function or np.ndarray or theano.shared
        :math:`b_o`
    nonlinearity_outgate : function
        :math:`\sigma`
    nonlinearity_out : function or np.ndarray or theano.shared
        :math:`\tanh`
    cell_init : function or np.ndarray or theano.shared
        :math:`c_0`
    hid_init : function or np.ndarray or theano.shared
        :math:`h_0`
    hid_init : function or np.ndarray or theano.shared
        If ouput_network is not None then this is used to initialize y
    backwards : boolean
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from x_1 to x_n.
    learn_init : boolean
        If True, initial hidden values are learned
    peepholes : boolean
        If True, the LSTM uses peephole connections.
        When False, W_cell_to_ingate, W_cell_to_forgetgate and
        W_cell_to_outgate are ignored.
    gradient_steps : int
        Number of timesteps to include in backpropagated gradient
        If -1, backpropagate through the entire sequence
    return_sequence : boolean
        Specifies if the LSTM should output the hidden state for all positions
        in the sequence or only for the last position. If true the output
        shape is (num_batch, sequence_lengt, num_units) if false the output
        shape is (num_batch, num_units).
    return_cell : boolean
        If true the cell and the hidden state is returned. This setting might
        be useful if you are implementing an encoder/decoder structure.
    grad_clipping: False or float
        If float the gradient messages are clipped during the backward pass.
        See [1]_ (p. 6) for further explanation.
    output_network : None or :class:`lasagne.layers.Layer`
        If not none then then the layer is used to calculate predictions
        that are used as input for the next time step in the LSTM.

    References
    ----------
    .. [1] Alex Graves : Generating Sequences With Recurrent Neural
           Networks
    .. [2] Wojciech Zaremba et al.,
           Recurrent neural networkregularization"
    """
    def __init__(self, incoming, num_units,
                 W_in_to_ingate=init.Normal(0.1),
                 W_hid_to_ingate=init.Normal(0.1),
                 W_cell_to_ingate=init.Normal(0.1),
                 b_ingate=init.Normal(0.1),
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 W_in_to_forgetgate=init.Normal(0.1),
                 W_hid_to_forgetgate=init.Normal(0.1),
                 W_cell_to_forgetgate=init.Normal(0.1),
                 b_forgetgate=init.Normal(0.1),
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 W_in_to_cell=init.Normal(0.1),
                 W_hid_to_cell=init.Normal(0.1),
                 b_cell=init.Normal(0.1),
                 nonlinearity_cell=nonlinearities.tanh,
                 W_in_to_outgate=init.Normal(0.1),
                 W_hid_to_outgate=init.Normal(0.1),
                 W_cell_to_outgate=init.Normal(0.1),
                 b_outgate=init.Normal(0.1),
                 W_y_to_ingate=init.Normal(0.1),
                 W_y_to_forgetgate=init.Normal(0.1),
                 W_y_to_cell=init.Normal(0.1),
                 W_y_to_outgate=init.Normal(0.1),
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 y_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 return_sequence=True,
                 return_cell=False,
                 grad_clipping=False,
                 output_network=None,
                 hid_init_val=None):

        # todo: add subnetwork option.

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incoming)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

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
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.return_sequence = return_sequence
        self.return_cell = return_cell
        self.grad_clipping = grad_clipping
        self.output_network = output_network
        self.hid_init_val = hid_init_val

        (self.num_batch, _, num_inputs) = self.input_shape

        # Initialize parameters using the supplied args
        self.W_in_to_ingate = self.add_param(
            W_in_to_ingate, (num_inputs, num_units), name="W_in_to_ingate")

        self.W_hid_to_ingate = self.add_param(
            W_hid_to_ingate, (num_units, num_units), name="W_hid_to_ingate")

        self.b_ingate = self.add_param(
            b_ingate, (num_units,), name="b_ingate", regularizable=False)

        self.W_in_to_forgetgate = self.add_param(
            W_in_to_forgetgate, (num_inputs, num_units),
            name="W_in_to_forgetgate")

        self.W_hid_to_forgetgate = self.add_param(
            W_hid_to_forgetgate, (num_units, num_units),
            name="W_hid_to_forgetgate")

        self.b_forgetgate = self.add_param(
            b_forgetgate, (num_units,), name="b_forgetgate",
            regularizable=False)

        self.W_in_to_cell = self.add_param(
            W_in_to_cell, (num_inputs, num_units), name="W_in_to_cell")

        self.W_hid_to_cell = self.add_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")

        self.b_cell = self.add_param(
            b_cell, (num_units,), name="b_cell", regularizable=False)

        self.W_in_to_outgate = self.add_param(
            W_in_to_outgate, (num_inputs, num_units), name="W_in_to_outgate")

        self.W_hid_to_outgate = self.add_param(
            W_hid_to_outgate, (num_units, num_units), name="W_hid_to_outgate")

        self.b_outgate = self.add_param(
            b_outgate, (num_units,), name="b_outgate", regularizable=False)

        # Stack input to gate weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_to_gates = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden to gate weight matrices
        self.W_hid_to_gates = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack gate biases into a (4*num_units) vector
        self.b_gates = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        # Initialize peephole (cell to gate) connections.  These are
        # elementwise products with the cell state, so they are represented as
        # vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                W_cell_to_ingate, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                W_cell_to_forgetgate, (num_units, ),
                name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                W_cell_to_outgate, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(
            cell_init, (1, num_units), name="cell_init",
            trainable=learn_init, regularizable=False, recurrent_init=True)
        self.hid_init = self.add_param(
            hid_init, (1, num_units), name="hid_init",
            trainable=learn_init, regularizable=False, recurrent_init=True)

        if self.output_network is not None:
            num_classes = helper.get_output_shape(self.output_network)[-1]
            self.y_init = self.add_param(
                y_init, (1, num_classes), name="y_init",
                trainable=learn_init, regularizable=False)

            self.W_y_to_ingate = self.add_param(
                W_y_to_ingate, (num_classes, num_units),
                name="W_y_to_ingate")

            self.W_y_to_forgetgate = self.add_param(
                W_y_to_forgetgate, (num_classes, num_units),
                name="W_y_to_forgetgate")

            self.W_y_to_cell = self.add_param(
                W_y_to_cell, (num_classes, num_units),
                name="W_y_to_celll")

            self.W_y_to_outgate = self.add_param(
                W_y_to_outgate, (num_classes, num_units),
                name="W_in_to_outgate")

            self.W_y_to_gates = T.concatenate(
                [self.W_y_to_ingate, self.W_y_to_forgetgate,
                 self.W_y_to_cell, self.W_y_to_outgate], axis=1)

    def get_params(self, **tags):
        params = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            params = [param for param in params
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            params = [param for param in params
                      if not (self.params[param] & exclude)]

        if self.output_network is not None:
            params += helper.get_all_params(self.output_network, **tags)

        return params

    def get_output_shape_for(self, input_shape):
        if self.output_network is None:
            num_outputs = self.num_units
        else:
            num_outputs = helper.get_output_shape(self.output_network)[-1]

        if self.return_sequence:
            return input_shape[0], input_shape[1], num_outputs
        else:
            return input_shape[0], num_outputs

    def get_output_for(self, input, mask=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If None,
            then it assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape (n_batch, n_time_steps) where
            `mask[i, j] = 1` when `j <= (length of sequence i)` and
            `mask[i, j] = 0` when `j > (length of sequence i)`.
        """
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Because the input is given for all time steps, we can precompute
        # the inputs to the gates before scanning.
        # input is dimshuffled to (n_time_steps, n_batch, n_features)
        # W_in_to_gates is (n_features, 4*num_units). input_dot_w is then
        # (n_time_steps, n_batch, 4*num_units).
        input_dot_w = T.dot(input, self.W_in_to_gates) + self.b_gates

        # input_dot_w is (n_batch, n_time_steps, 4*num_units). We define a
        # slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_dot_w_n is the nth timestep of the input, dotted with W
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        def step(input_dot_w_n, cell_previous, hid_previous):
            # Calculate gates pre-activations and slice
            gates = input_dot_w_n + T.dot(hid_previous, self.W_hid_to_gates)

            # clip gradients
            if self.grad_clipping is not False:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity_out(cell)

            if self.output_network is None:
                return [cell, hid]
            else:
                y = helper.get_output(self.output_network, hid, **kwargs)
                return [cell, hid, y]

        def step_masked(input_dot_w_n, mask_n, cell_previous, hid_previous):

            cell, hid = step(input_dot_w_n, cell_previous, hid_previous)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask = 1 - mask_n
            cell = cell*mask_n + cell_previous*not_mask
            hid = hid*mask_n + hid_previous*not_mask

            return [cell, hid]

        def step_feedback(input_dot_w_n, cell_previous, hid_previous,
                          y_previous):
            input_dot_w_n += T.dot(y_previous, self.W_y_to_gates)
            cell, hid, y = step(input_dot_w_n, cell_previous, hid_previous)

            return [cell, hid, y]

        def step_feedback_masked(input_dot_w_n, mask_n, cell_previous,
                                 hid_previous, y_previous):
            input_dot_w_n += T.dot(y_previous, self.W_y_to_gates)
            cell, hid, y = step(input_dot_w_n, cell_previous, hid_previous)
            not_mask = 1 - mask_n
            cell = cell*mask_n + cell_previous*not_mask
            hid = hid*mask_n + hid_previous*not_mask
            y = y*mask_n + y_previous*mask_n

            return [cell, hid, y]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input_dot_w, mask]

            if self.output_network is None:
                step_fun = step_masked
            else:
                step_fun = step_feedback_masked
        else:
            sequences = input_dot_w

            if self.output_network is None:
                step_fun = step
            else:
                step_fun = step_feedback

        # repeat cell and hid init to batch size
        ones = T.ones((self.num_batch, 1))


        if self.hid_init_val is None:
            if self.num_batch > 1:
                hid_init = T.dot(ones, self.hid_init)
                cell_init = T.dot(ones, self.cell_init)
            else:
                hid_init = self.hid_init
                cell_init = self.cell_init
            init = [hid_init, cell_init]
        else:
            init = self.hid_init_val

        if self.output_network is not None:
            init += [T.dot(ones, self.y_init)]

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        output = theano.scan(
            step_fun,
            sequences=sequences,
            outputs_info=init,
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps)[0]

        if self.output_network is None:
            cell_out, hid_out = output
        else:
            cell_out, _, hid_out = output


        self.cell = cell_out.dimshuffle(1, 0, 2)
        self.hid = hid_out.dimshuffle(1, 0, 2)

        if self.return_sequence:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1, :]
        else:
            # no need to dimshuffle because we return the last state
            hid_out = hid_out[-1]
        output = hid_out


        return output

    def get_hidden_values(self):
        return self.cell, self.hid


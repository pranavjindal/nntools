"""
Nonlinearities
"""
import theano
import theano.tensor as T

import lasagne
from lasagne import init

# sigmoid
from theano.tensor.nnet import sigmoid

# softmax (row-wise)
from theano.tensor.nnet import softmax

# tanh
from theano.tensor import tanh

# rectify
# The following is faster than lambda x: T.maximum(0, x)
# Thanks to @SnippyHolloW for pointing this out.
# See: https://github.com/SnippyHolloW/abnet/blob/807aeb98e767eb4e295c6d7d60ff5c9006955e0d/layers.py#L15
def rectify(x):
    return (x + T.abs_(x)) / 2 #  ~ sec 4.6
    #return T.max(.0,x)      # ~ sec 4.6
    #return x*(x > 0)
    #return T.switch(x < 0., 0., x)

def rectify_capped(cap):
    return lambda x:T.minimum(rectify(x), cap )

def rectify_leaky(leakyness):
    assert leakyness < 1 and leakyness > 0, "leakyness should be ]0-1["
    return lambda x: T.maximum(x,x*leakyness)

def rectify_leaky_capped(cap,leakyness):
    f_leaky = rectify_leaky(leakyness)
    return lambda x: T.minimum(f_leaky(x), cap)

def LAunit(
        num_units,segments=5, init_a=init.Uniform((-0.05, 0.05)),
        init_b=init.Uniform((-0.05, 0.05)),
        relucap=0):
    '''
        http://arxiv.org/abs/1412.6830
    '''
    a = theano.shared(init_a(shape=(num_units, segments)))
    b = theano.shared(init_b(shape=(num_units, segments)))

    # cap all segments to relucap / segments
    if relucap > 0:
        rec = rectify_capped(float(relucap)/segments)
    else:
        rec = rectify

    def la_act(act):
        return rec(act) + T.sum(rec(-act.dimshuffle(0, 1, 'x') +
                                   b.dimshuffle('x', 0, 1))*a.dimshuffle('x', 0, 1),
                                   axis=2)

    return la_act, a, b




# linear
def linear(x):
    return x

identity = linear

This code was used to produce the results in http://arxiv.org/abs/1412.7828

The code is a fork of work by Colin Raffel (https://github.com/craffel).

The main contributions are:

*  Faster LSTM/RNN implementation
*  Bidirectional LSTM/RNN
*  LSTM Dropout  (http://arxiv.org/abs/1409.2329)
*  Setupfunctions
*  Learned activation units + other activation funtions (http://arxiv.org/abs/1412.6830)

Install by cloning this repository and install with:

```PYTHON
python setup.py develop
```

from within the repository.

You'll need to install the development version of theano:

http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions

Run:

```PYTHON
python examples/lstm_long.py
python examples/lstm_shor.py
```

for example code.

Lasagne
=======

Lasagne is a lightweight library to build and train neural networks in Theano.

Lasagne is a work in progress, input is welcome.

Design goals:

* Simplicity: it should be easy to use and extend the library. Whenever a feature is added, the effect on both of these should be considered. Every added abstraction should be carefully scrutinized, to determine whether the added complexity is justified.

* Small interfaces: as few classes and methods as possible. Try to rely on Theano's functionality and data types where possible, and follow Theano's conventions. Don't wrap things in classes if it is not strictly necessary. This should make it easier to both use the library and extend it (less cognitive overhead).

* Don't get in the way: unused features should be invisible, the user should not have to take into account a feature that they do not use. It should be possible to use each component of the library in isolation from the others.

* Transparency: don't try to hide Theano behind abstractions. Functions and methods should return Theano expressions and standard Python / numpy data types where possible.

* Focus: follow the Unix philosophy of "do one thing and do it well", with a strong focus on feed-forward neural networks.

* Pragmatism: making common use cases easy is more important than supporting every possible use case out of the box.

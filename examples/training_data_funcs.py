def one_hot(labels, n_classes):
    '''
    Converts an array of label integers to a one-hot matrix encoding

    :parameters:
        - labels : np.ndarray, dtype=int
            Array of integer labels, in {0, n_classes - 1}
        - n_classes : int
            Total number of classes

    :returns:
        - one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
            One-hot matrix of the input
    '''
    one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
    one_hot[range(labels.shape[0]), labels] = True
    return one_hot


def load_netcdf(filename):
    '''
    Loads in data from a netcdf file in rnnlib format

    :parameters:
        - filename : str
            Path to a netcdf file

    :returns:
        - X : list of np.ndarray
            List of time series matrices
        - y : list of np.ndarray
            List of label arrays in one-hot form (see one_hot)
    '''
    with open(filename, 'r') as f:
        netcdf_data = scipy.io.netcdf_file(f).variables

    X = []
    y = []
    n = 0
    for length in netcdf_data['seqLengths'].data:
        X_n = netcdf_data['inputs'].data[n:n + length]
        X.append(X_n.astype(theano.config.floatX))
        y_n = one_hot(netcdf_data['targetClasses'].data[n:n + length],
                      netcdf_data['numTargetClasses'].data)
        y.append(y_n.astype(theano.config.floatX))
        n += length
    return X, y


def make_batches(X, length, batch_size=BATCH_SIZE):
    '''
    Convert a list of matrices into batches of uniform length

    :parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            matrix
    '''
    n_batches = len(X)//batch_size
    X_batch = np.zeros((n_batches, batch_size, length, X[0].shape[1]),
                       dtype=theano.config.floatX)
    X_mask = np.zeros(X_batch.shape, dtype=np.bool)
    for b in range(n_batches):
        for n in range(batch_size):
            X_m = X[b*batch_size + n]
            X_batch[b, n, :X_m.shape[0]] = X_m[:length]
            X_mask[b, n, :X_m.shape[0]] = 1
    return X_batch, X_mask
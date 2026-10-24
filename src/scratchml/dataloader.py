import gzip

import numpy as np


# Loads Fashion MNIST from saved archives, which are required to be present in
# the project tree. Returns four values: X_train, Y_train, X_test, Y_test.
#
# The X values are arrays of input images. The images are by default represented
# as 28x28 matrices of integers between 0 and 255. If flatten=True, they are
# represented as arrays of length 784. If normalize=True, they are represented
# using floats ranging from 0.0 to 1.0.
#
# The Y values are 
def load_fashion_mnist(flatten=False, normalize=False, vectorize_output=False):
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    
    # Load training images.
    with gzip.open("data/fashionmnist/train-images-idx3-ubyte.gz", "rb") as f:
        X_train = (
            # Offset=16 skips the magic number, num_imags, num_rows, num_columns
            np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            .copy()
            .reshape(-1, 28, 28)
            .astype(np.float32)
        )
    
    # Load training labels
    with gzip.open("data/fashionmnist/train-labels-idx1-ubyte.gz", "rb") as f:
        Y_train = (
            # Offset=8 skips the magic number, and the number of items
            np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            .copy()
            .astype(np.int32)
        )
    
    # Load test images.
    with gzip.open("data/fashionmnist/t10k-images-idx3-ubyte.gz", "rb") as f:
        X_test = (
            np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            .copy()
            .reshape(-1, 28, 28)
            .astype(np.float32)
        )
    
    # Load test labels
    with gzip.open("data/fashionmnist/t10k-labels-idx1-ubyte.gz", "rb") as f:
        Y_test = (
            # Offset=8 skips the magic number, and the number of items
            np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            .copy()
            .astype(np.int32)
        )
    
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    
    if vectorize_output:
        Y_train = np.eye(10)[Y_train]
        Y_test = np.eye(10)[Y_test]
    
    return X_train, Y_train, X_test, Y_test


def downsample(*arrays, size, rng=None):
    """
    Given m datasets each of size n (that is, m arrays with the same outer
    dimension n), this function draws a random subpermutation of length k
    from 0..=n-1 and returns the corresponding elements of each dataset.

    >>> X_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> Y_train = X_train**2
    >>> draw_sample(X_train, Y_train, 1)
    ([2.0, 4.0], [4.0, 16.0])
    """
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays must have same outer dimension")

    if (rng == None):
        rng = np.random.default_rng(seed=2345436)

    idx = rng.choice(n, size=size, replace=False)

    return tuple(np.asarray(a)[idx] for a in arrays)

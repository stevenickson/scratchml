import numpy as np

from ._initializer import Initializer


class HeNormal(Initializer):
    """
    He initialization creates NumPy arrays whose entries are sampled from
    a normal distribution with variance 2/n_k, where n_k is the
    number of inputs to a given neuron. In a fully connected layer, n_k
    is the input dimension.
    """


    def __init__(self, rng):
        """Create a He initializer function.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator to use for pseudorandom number generation.
        """

        self.rng = rng


    def __call__(self, shape):
        """Create a random array using He initialization.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array to be generated. The input dimension
            is taken to be shape[0], matching the X @ W layout used by
            Product.

        Returns
        -------
        numpy.ndarray
            A random numpy array of the given shape whose elements are sampled from
            a He normal distribution.
        """

        return self.rng.normal(
            loc=0.0,
            scale=(np.sqrt(2 / shape[0])),
            size=shape
        )

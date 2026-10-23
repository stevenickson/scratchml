import numpy as np

from ._initializer import Initializer


class Glorot(Initializer):
    """
    Glorot initialization creates NumPy arrays whose entries are sampled from
    a uniform distribution with low value -r and high value r, where r is

    sqrt(6 / (m_k + n_k))

    where m_k is the number of inputs to a given neuron and n_k is the number
    of outputs. In a fully connected layer, m_k is the input dimension and
    n_k is the output dimension.
    """


    def __init__(self, rng):
        """Create a Glorot initializer function.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator to use for pseudorandom number generation.
        """

        self.rng = rng


    def __call__(self, shape):
        """Create a random matrix using Glorot initialization.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array to be generated. The input dimension
            is taken to be shape[0], matching the X @ W layout used by
            Product. Similarly, the output dimension is shape[1].

        Returns
        -------
        numpy.ndarray
            A random numpy matrix of the given shape whose elements are sampled
            from a Glorot uniform distribution. The input shape must be 2D.
        """

        r = np.sqrt(6.0 / (shape[0] + shape[1]))

        return self.rng.uniform(
            low=-r,
            high=r,
            size=shape
        )

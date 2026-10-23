import numpy as np

from ._initializer import Initializer


class Trivial(Initializer):
    """The trivial initializer creates a NumPy array filled with zeros."""


    def __init__(self):
        pass


    def __call__(self, shape):
        """Create an array of zeros, represented as a NumPy array.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array to be generated.

        Returns
        -------
        numpy.ndarray
            A numpy array of the given shape whose entries are zeros.
        """

        return np.zeros(shape)

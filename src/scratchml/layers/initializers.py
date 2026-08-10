"""Initialize machine learning models.

This module contains a series of classes designed to create
NumPy matrices filled with random values. Much like the rest of
this framework, these classes are intended to be treated as instantiable
functions. Parameters such as seed, mean, and variance are chosen
at the time of instantiation. The input and output dimensions
of the final matrix are passed as arguements to the __call__ function
of each class.
"""

import abc

import numpy as np


class MatrixInitializer(abc.ABC):
    """
    Superclass for matrix initializer types. Subclasses must
    implement __call__, which must accept an input dimension and
    and output dimension as arguements. The output of __call__ is
    expected to be a (dim_in, dim_out) numpy array.
    """


    @abc.abstractmethod
    def __call__(self, dim_in, dim_out) -> np.ndarray:
        """Create a random matrix.

        Parameters
        ----------
        dim_in : int
            The input dimension of the matrix to be generated.
        dim_out : int
            The output dimension of the matrix to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of shape (dim_in, dim_out)
        """
        ...


class Trivial(MatrixInitializer):
    """The trivial initializer creates a NumPy array filled with zeros."""


    def __init__(self):
        pass


    def __call__(self, dim_in, dim_out):
        """Create zero matrix, represented as a NumpPy Array.

        Parameters
        ----------
        dim_in : int
            The input dimension of the matrix to be generated.
        dim_out : int
            The output dimension of the matrix to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of shape (dim_in, dim_out) whose are zeros.
        """

        return np.zeros((dim_in, dim_out))


class Uniform(MatrixInitializer):
    """
    The uniform initializer creates a matrix of the desired shape whose elements are
    sampled from a uniform distribution.
    """


    def __init__(self, seed=0, low=-0.5, high=0.5):
        """Create a uniform initializer function.

        Parameters
        ----------
        seed : int
            The seed for pseudorandom number generation
        low : float
            The lowest value of the distribution to sample from
        high : float
            The highest value of the distribution to sample from
        """

        self.rng = np.random.default_rng(seed=seed)
        self.low = low
        self.high = high


    def __call__(self, dim_in, dim_out):
        """Create a random matrix whose entries are uniformly distributed.

        Parameters
        ----------
        dim_in : int
            The input dimension of the matrix to be generated.
        dim_out : int
            The output dimension of the matrix to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of shape (dim_in, dim_out) whose elements are sampled from
            a uniform distribution.
        """

        return self.rng.uniform(
            low=self.low,
            high=self.high,
            size=(dim_in, dim_out)
        )


class Normal(MatrixInitializer):
    """
    The normal initializer creates a matrix of the desired shape whose elements are
    sampled from a uniform distribution.
    """


    def __init__(self, seed=0, mean=0, variance=0.5):
        """Create a normal initializer function.

        Parameters
        ----------
        seed : int
            The seed to use for pseudorandom number generation.
        mean: float
            The mean of the normal distribution from which the entries are to
            be sampled.
        variance: float
            The variance of the normal distribution from which the entries
            are to be sampled
        """

        self.rng = np.random.default_rng(seed=seed)
        self.mean = mean
        self.variance = variance


    def __call__(self, dim_in, dim_out):
        """Create a random matrix using a normal distribution.

        Parameters
        ----------
        dim_in : int
            The input dimension of the matrix to be generated.
        dim_out : int
            The output dimension of the matrix to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of shape (dim_in, dim_out) whose elements are sampled from
            a normal distribution.
        """

        return self.rng.normal(
            loc=self.mean,
            scale=self.variance,
            size=(dim_in, dim_out)
        )


class HeNormal(MatrixInitializer):
    """
    He initialization creates matrices whose entries are sampled from
    a normal distribution with with variance 2/n_k, where n_k is the
    number of inputs to a given neuron. In a fully conneted layer, n_k
    is the input dimension.
    """


    def __init__(self, seed=0, mean=0):
        """Create a He initializer function.

        Parameters
        ----------
        seed : int
            The seed to use for pseudorandom number generation.
        mean: float
            The mean of the normal distribution from which the entries are to
            be sampled.
        """

        self.rng = np.random.default_rng(seed=seed)
        self.mean = mean


    def __call__(self, dim_in, dim_out):
        """Create a random matrix using He initialization.

        Parameters
        ----------
        dim_in : int
            The input dimension of the matrix to be generated.
        dim_out : int
            The output dimension of the matrix to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of shape (dim_in, dim_out) whose elements are sampled from
            a He normal distribution.
        """

        return self.rng.normal(
            loc=self.mean,
            scale=(np.sqrt(2 / dim_in)),
            size=(dim_in, dim_out)
        )

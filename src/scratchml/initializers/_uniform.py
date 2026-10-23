from ._initializer import Initializer


class Uniform(Initializer):
    """
    The uniform initializer creates an array of the desired shape whose elements
    are sampled from a uniform distribution.
    """


    def __init__(self, rng, low=-0.5, high=0.5):
        """Create a uniform initializer function.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator to use for pseudorandom number generation
        low : float
            The lowest value of the distribution to sample from
        high : float
            The highest value of the distribution to sample from
        """

        self.rng = rng
        self.low = low
        self.high = high


    def __call__(self, shape):
        """Create a random array whose entries are uniformly distributed.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of the given shape whose elements are sampled from
            a uniform distribution.
        """

        return self.rng.uniform(
            low=self.low,
            high=self.high,
            size=shape
        )

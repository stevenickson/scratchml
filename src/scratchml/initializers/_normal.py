from ._initializer import Initializer


class Normal(Initializer):
    """
    The normal initializer creates an array of the desired shape whose elements are
    sampled from a normal distribution.
    """


    def __init__(self, rng, mean=0, standard_deviation=0.5):
        """Create a normal initializer function.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator to use for pseudorandom number generation.
        mean: float
            The mean of the normal distribution from which the entries are to
            be sampled.
        standard_deviation: float
            The standard deviation of the normal distribution from which the
            entries are to be sampled
        """

        self.rng = rng
        self.mean = mean
        self.standard_deviation = standard_deviation


    def __call__(self, shape):
        """Create a random array using a normal distribution.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array to be generated.

        Returns
        -------
        numpy.ndarray
            A random numpy array of the given shape whose elements are sampled from
            a normal distribution.
        """

        return self.rng.normal(
            loc=self.mean,
            scale=self.standard_deviation,
            size=shape
        )

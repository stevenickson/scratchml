import abc


class Initializer(abc.ABC):
    """
    Superclass for initializer types. Subclasses must
    implement __call__, which must accept the shape of the array
    to be generated. The output of __call__ is expected to be a
    numpy array of that shape.
    """


    @abc.abstractmethod
    def __call__(self, shape):
        """Create a random array.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array to be generated. For a weight
            matrix, this is (dim_in, dim_out).

        Returns
        -------
        numpy.ndarray
            A random numpy array of the given shape
        """
        ...

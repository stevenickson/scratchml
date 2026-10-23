"""Initialize machine learning models.

This module contains a series of classes designed to create
NumPy arrays filled with random values. Much like the rest of
this framework, these classes are intended to be treated as instantiable
functions. Parameters such as the random generator, mean, and standard
deviation are chosen at the time of instantiation. The shape of the final
array is passed as an argument to the __call__ function of each class.
"""

from ._glorot import Glorot
from ._he_normal import HeNormal
from ._initializer import Initializer
from ._normal import Normal
from ._trivial import Trivial
from ._uniform import Uniform

__all__ = [
        'Glorot',
        'HeNormal',
        'Initializer',
        'Normal',
        'Trivial',
        'Uniform'
]

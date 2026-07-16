import numpy as np

class LinearTransformation:
    def __init__(self, m):
        self.W = m


    def __init__(self, dimension_in, dimension_out, initializer):
        self.W = initializer(dimension_in, dimension_out)
        self.W = self.rng.normal(0, np.sqrt(2 / 784), size=(dimension_in, dimension_out))


    def __call__(self, x):
        return x @ self.W


    def jacobian(self, x):
        return self.W


class Bias:
    def __init__(self, dimension):
        self.b = np.zeros((1, dimension))


    def __call__(self, x):
        return x + self.b


    def jacobian(self, x):
        return np.identity(self.dimension)
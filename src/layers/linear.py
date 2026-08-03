import numpy as np

class LinearTransformation:
    def __init__(self, weights):
        self.W = weights


    def __init__(self, dimension_in, dimension_out, initializer):
        self.W = initializer(dimension_in, dimension_out)


    def __call__(self, x):
        return x @ self.W


    def jacobian(self, x):
        return self.W


class Bias:
    def __init__(self, weights):
        self.b = weights

        
    def __init__(self, dimension, initializer):
        self.b = initializer(1, dimension)


    def __call__(self, x):
        return x + self.b


    def jacobian(self, x):
        return np.identity(self.dimension)
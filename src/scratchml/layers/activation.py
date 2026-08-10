import numpy as np


class ReLU:
    def __init__(self, leak=0.0):
        self.leak = leak


    def __call__(self, x):
        return np.where(x < 0, self.leak * x, x)


    def jacobian(self, x):
        return np.where(x < 0, self.leak, 1.0)

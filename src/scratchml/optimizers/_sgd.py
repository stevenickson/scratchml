import numpy as np


class SGD:
    def __init__(self, loss, parameters, rate=0.15, beta=0.0):
        self.rate = rate
        self.beta = beta
        self.parameters = parameters

        self.gradients = [np.zeros_like(p.evaluation) for p in parameters]


    def step(self):
        for parameter, gradient in zip(self.parameters, self.gradients):
            gradient *= self.beta
            gradient += parameter.backpropagation * self.rate
            parameter.evaluation -= gradient


import numpy as np

from ._node import Node


class MSE(Node):
    def __init__(self, x, target):
        self.x = x
        self.target = target
        self.evaluation = None
        self.x_derivative = None
        self.target_derivative = None
        self.backpropagation = np.array([0.0])


    def get_parents(self):
        return (self.x, self.target)


    def forwards(self):
        a = self.x.evaluation
        b = self.target.evaluation
        self.evaluation = np.mean((b - a)**2)
        self.x_derivative = (2 / a.size) * (a - b)
        self.target_derivative = (2 / a.size) * (b - a)


    def back(self):
        self.x.backpropagation += self.backpropagation * self.x_derivative
        self.target.backpropagation += self.backpropagation * self.target_derivative

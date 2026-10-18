import numpy as np

from ._node import Node


class Tanh(Node):
    def __init__(self, x):
        self.x = x
        self.evaluation = None
        self.backpropogation = None


    def get_parents(self):
        return (self.x,)


    def forwards(self):
        self.evaluation = np.tanh(self.x.evaluation)


    def back(self):
        self.x.backpropogation += self.backpropogation * (1 - self.evaluation**2)

import numpy as np

from ._node import Node


class Constant(Node):
    def __init__(self, value):
        self.evaluation = value
        self.backpropagation = np.array([0.0])


    def get_parents(self):
        return ()


    def forwards(self):
        pass


    def back(self):
        pass


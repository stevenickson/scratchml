import numpy as np

from ._node import Node
from ._shape import reduce_to


class Sum(Node):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.evaluation = None
        self.backpropagation = np.array([0.0])


    def get_parents(self):
        return (self.x1, self.x2)


    def forwards(self):
        self.evaluation = self.x1.evaluation + self.x2.evaluation


    def back(self):
        self.x1.backpropagation += reduce_to(
                self.backpropagation,
                np.shape(self.x1.evaluation)
        )
        self.x2.backpropagation += reduce_to(
                self.backpropagation,
                np.shape(self.x2.evaluation)
        )

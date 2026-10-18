import numpy as np

from ._node import Node


class ReLU(Node):
    def __init__(self, x, alpha=0.0):
        self.x = x
        self.alpha = alpha
        self.evaluation = None
        self.derivative = None
        self.backpropogation = np.array([0.0])


    def get_parents(self):
        return (self.x,)


    def forwards(self):
        #self.evaluation = np.maximum(0, self.x.evaluation)
        # self.derivative = (np.asarray(self.x.evaluation) > 0).astype(float)
        self.evaluation = np.where(self.x.evaluation > 0, self.x.evaluation, self.x.evaluation * self.alpha)
        self.derivative = np.where(self.x.evaluation > 0, 1.0, self.alpha)


    def back(self):
        self.x.backpropogation += self.backpropogation * self.derivative

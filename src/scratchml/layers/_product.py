import numpy as np

from ._node import Node


class Product(Node):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.evaluation = None
        self.x1_derivative: np.ndarray | None = None
        self.x2_derivative: np.ndarray | None = None
        self.backpropagation = np.array([0.0])


    def get_parents(self):
        return (self.x1, self.x2)


    def forwards(self):
        self.evaluation = self.x1.evaluation @ self.x2.evaluation
        self.x1_derivative = self.x2.evaluation.T
        self.x2_derivative = self.x1.evaluation


    def back(self):
        self.x1.backpropagation += self.backpropagation @ self.x1_derivative
        self.x2.backpropagation += self.x2_derivative.T @ self.backpropagation

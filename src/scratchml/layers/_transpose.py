import numpy as np

from ._node import Node


class Transpose(Node):
    def __init__(self, x):
        self.x = x
        self.evaluation = None
        self.backpropogation = np.array([0.0])


    def get_parents(self):
        return (self.x,)


    def forwards(self):
        # Strictly speaking, this node can be viewed as applying a transpose
        # identity matrix to the input, which would have the effect of
        # transposing the input. However, numpy allows us to take transposes
        # without allocating memory for an identity matrix, so we take
        # advantage of it, both here and in the backwards pass.
        self.evaluation = self.x.evaluation.T


    def back(self):
        # As noted in the forward pass, this node can be viewed as a simple
        # multiplication by a constant rectangular identity matrix. As a result,
        # the derivative is just that matrix. We can simplify the backward pass
        # expression by noting that this matrix has the effect of transposing
        # its input. Therefore, the backflow is just the tranpose of the
        # incoming backflow.
        self.x.backpropogation += self.backpropogation.T

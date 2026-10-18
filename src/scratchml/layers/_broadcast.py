import numpy as np

from ._node import Node


class Broadcast(Node):
    def __init__(self, x, like):
        self.x = x
        self.like = like
        self.evaluation = None
        self.backpropogation = None


    def get_parents(self):
        return (self.x, self.like)


    def forwards(self):
        self.evaluation = np.broadcast_to(self.x.evaluation, self.like.evaluation.shape)


    def back(self):
        mismatched_axes = tuple(i for i, (p, o) in
                                    enumerate(
                                        zip(
                                            self.x.evaluation.shape,
                                            self.backpropogation.shape
                                        )
                                    )
                                if p == 1 and o > 1)
        self.x.backpropogation += self.backpropogation.sum(
                axis=mismatched_axes,
                keepdims = True
        )

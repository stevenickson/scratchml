from collections import deque

import numpy as np


class Node:
    def __init__(self):
        self.evaluation = None

    def get_parents(self):
        ...


    def forwards(self):
        ...


    def back(self):
        ...


    def toposort(self):
        degrees = {self: 0}
        frontier = [self]

        # Explore the graph and note the number of incoming connections
        # in the "degrees" dictionary
        while frontier:
            ptr = frontier.pop()
            for p in ptr.get_parents():
                if p in degrees:
                    degrees[p] += 1
                else:
                    degrees[p] = 1
                    frontier.append(p)

        # Next, peel off degree zero nodes and put them first in the order.
        # Decrement the order of their parents. Repeat until all nodes
        # have been visited.
        queue = deque(node for node, degree in degrees.items() if degree == 0)
        ordering = []
        while queue:
            u = queue.popleft()
            ordering.append(u)
            for p in u.get_parents():
                degrees[p] -= 1
                if degrees[p] == 0:
                    queue.append(p)

        if len(ordering) != len(degrees):
            raise ValueError("Input graph is cyclic")

        return ordering


    def __call__(self):
        for n in reversed(self.toposort()):
            n.forwards()


    def backpropogate(self):
        ordering = self.toposort()
        for n in ordering:
            n.backpropogation = np.zeros_like(n.evaluation)
        self.backpropogation = np.ones_like(self.evaluation)

        for n in ordering:
            n.back()

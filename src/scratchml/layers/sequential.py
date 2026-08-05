import numpy as np

class Sequential():
    def __init__(self, *args, **kwargs):
        self.layers = []
        for x in args:
            self.layers += [x]


    def __call__(self, x):
        # Recursively apply each layer to the output of the
        # previous layer
        y = x
        for l in self.layers:
            y = l(y)
        return y


    def jacobian(self, x):
        # Returns the Jacobian of the Sequential at x
        d = x
        for l in layers.reverse():
            d = l.jacobian(l(d)) @ l.jacobian(d)
        return d
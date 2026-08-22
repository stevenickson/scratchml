class Sequential:
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
        # Naive implementation of forward-mode autodifferentiation.
        # Forward mode differentiation works like this: beginning with
        # the first layer, the cumulative derivative is
        #     cumulative_derivative @ layer_derivative(cumulative_output)
        # per the chain rule.
        d = 0
        y = 0
        for l in self.layers[:1]:
            d = l.jacobian(x)
            y = l(x)

        for l in self.layers[1:]:
            d = d @ l.jacobian(y)
            y = l(y)

        return d

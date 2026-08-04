"""Multi-layer perceptrons.

This module implements the Perceptron, both single and multi-layer.
"""

import numpy as np
import linear
import activation
import sequential
import initializers

class Perceptron():
    def __init__(
        self,
        input_features,
        output_features,
        initializer,
        activation_function=activation.ReLU()
    ):
        # A perceptron is a sequence of a matrix multiplication,
        # a vector addition, and an activation  
        self.sequence = sequential.Sequential(
            linear.LinearTransformation(input_features, output_features, initializer),
            linear.Bias(output_features, initializer),
            activation=activation_function
        )


    def __call__(self, x):
        return self.sequence(x)


    def jacobian(self, x):
        return self.sequence.jacobian(x)


class MultilayerPerceptron():
    def __init__(
        self,
        layer_dimensions,
        initializer,
        activation_function
    ):
        # layer_dimensions is a list of the form [1, 3, 5, 3].
        # Each number is the dimension of a feature vector - this
        # class puts perceptrons between them.
        l = []
        for i in range(0, len(layer_dimensions) - 1):
            l.append(
                Perceptron(layer_dimensions[i],
                    layer_dimensions[i + 1],
                    initializer,
                    activation_function
                )
            )

        self.seq = sequential.Sequential(*l)


    def __call__(self, x):
        return self.seq(x)


    def jacobian(self, x):
        return self.seq.jacobian(x)
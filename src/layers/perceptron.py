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
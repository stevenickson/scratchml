import models.differentiable_model as dm
import numpy as np

class MiniModel(dm.DifferentiableModel):
    def __init__(self):
        self.randomizer = np.random.default_rng(seed=32142456)
        #self.W_1 = np.array([[1, 0, 0], randomizer.uniform(low=0.0, high=1, size=(3))])
        #self.W_2 = randomizer.uniform(low=0.0, high=1, size=(1, 2))
        self.W_1 = np.array([[1, 0, 0], [0.4, -0.2, 0.1]])
        self.W_2 = np.array([0.1, 0.2])

    def inference(self, input):
        r = input
        r = self.W_1.dot(r)
        r = np.tanh(r)
        r = self.W_2.dot(r)
        r = np.tanh(r)
        return r

    def loss(self, input):
        return inference(input)

    def loss_gradient(self, input):
        pass
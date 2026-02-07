import models.differentiable as dm
import numpy as np

class MiniModel(dm.DifferentiableModel):
    def __init__(self):
        self.randomizer = np.random.default_rng(seed=32142456)
        #self.W_1 = np.array([[1, 0, 0], randomizer.uniform(low=0.0, high=1, size=(3))])
        #self.W_2 = randomizer.uniform(low=0.0, high=1, size=(1, 2))
        self.W_1 = np.array([[1, 0, 0], [0.4, -0.2, 0.1]])
        self.W_2 = np.array([0.1, 0.2])

    
    def set_parameters(self, p):
        self.W_1[1][0] = p[0]
        self.W_1[1][1] = p[1]
        self.W_1[1][2] = p[2]
        self.W_2[0] = p[3]
        self.W_2[1] = p[4]

    
    def get_parameters(self):
        return np.array([
            self.W_1[1][0],
            self.W_1[1][1],
            self.W_1[1][2],
            self.W_2[0],
            self.W_2[1]])
        

    def inference(self, input):
        r = input
        r = self.W_1.dot(r)
        r = np.tanh(r)
        r = self.W_2.dot(r)
        r = np.tanh(r)
        return r


    def loss(self, input):
        return abs(self.inference(input), 2)


    def loss_gradient(self, input):
        # Cache inference steps for use in backpropogation
        u3 = self.W_1.dot(input)
        u2 = np.array([[u3[0][0]], [np.tanh(u3[1][0])]])
        u1 = self.W_2.dot(u2)
        l = np.tanh(u1)

        # Result of applying chain rule to derivative of the loss with
        # respect to weights
        e1 = 1 - pow(np.tanh(self.W_1[1][0] * input[0][0] + self.W_1[1][1] * input[1][0] + self.W_1[1][2] * input[2][0]), 2)
        e2 = input[1][0] * e1
        e3 = input[2][0] * e2
        du2 = np.array([[0, 0, 0, 0, 0],
                        [e1, e2, e3, 0, 0]])

        du1 = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 1]]).dot(u2) + np.array([self.W_2[0], self.W_2[1]]).dot(du2)
        dl = np.zeros((5, 5))
        np.fill_diagonal(dl, 1 - np.tanh(u1) * np.tanh(u1))
        dl[0][0] = 0
        dl = dl.dot(du1)
        return dl.dot(self.get_parameters())
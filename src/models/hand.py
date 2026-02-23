import models.differentiable as dm
import numpy as np

class HandModel(dm.DifferentiableModel):
    def __init__(self):
        self.W_1 = np.array([[0.2], [-0.1], [0.6]])
        self.b_1 = np.array([[0], [0.1], [0]])
        
        self.W_2 = np.array([[0.9, 0.4, 0.8], [0.6, 0.7, -0.4], [-0.7, 0.7, 0.7]])
        self.b_2 = np.array([[0.1], [0.1], [-0.4]])

        self.W_3 = np.array([[0.1, -0.4, 0.3], [0.6, -0.6, 0.9]])
        self.b_3 = np.array([[0.8], [-0.8]])

    def inference(self, x):
        z1 = self.W_1.dot(x) + self.b_1
        a1 = np.tanh(z1)
        z2 = self.W_2.dot(a1) + self.b_2
        a2 = np.tanh(z2)
        z3 = self.W_3.dot(a2) + self.b_3
        a3 = np.tanh(z3)
        return a3

    def loss(self, x):
        y = np.array([[x[0][0] * 0.5], [x[0][0] * 0.5]])
        a = self.inference(x)
        return 0.5 * np.transpose(a - y).dot(a - y)

    def loss_gradient(self, x):
        z1 = self.W_1.dot(x) + self.b_1
        a1 = np.tanh(z1)
        z2 = self.W_2.dot(a1) + self.b_2
        a2 = np.tanh(z2)
        z3 = self.W_3.dot(a2) + self.b_3
        a3 = np.tanh(z3)

        delta3 = ((a3 - np.array([[0.5 * x[0][0]], [0.5 * x[0][0]]])) * (1 - np.tanh(z3) * np.tanh(z3))).transpose()
        delta2 = delta3 @ self.W_3 * (1 - np.tanh(z2) * np.tanh(z2)).transpose()
        delta1 = delta2.dot(self.W_2) * (1 - np.tanh(z1) * np.tanh(z1)).transpose()

        lw1 = delta1.transpose() @ x
        lw2 = delta2.transpose() @ a1.transpose()
        lw3 = delta3.transpose() @ a2.transpose()
        lb1 = delta1
        lb2 = delta2
        lb3 = delta3
        
        return [lw1, lb1, lw2, lb2, lw3, lb3]

    def get_parameters():
        return [
            self.W_1,
            self.b_1,
            self.W_2,
            self.b_2,
            self.W_3,
            self.b_3
        ]

    def set_parameters(parameters):
        self.W_1 = parameters[0]
        self.b_1 = parameters[1]
        self.W_2 = parameters[2]
        self.b_2 = parameters[3]
        self.W_3 = parameters[4]
        self.b_3 = parameters[5]
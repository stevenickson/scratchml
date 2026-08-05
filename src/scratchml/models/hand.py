import numpy as np

class HandModel():
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

    def loss_gradient(self, xs):
        result = (0, 0, 0, 0, 0, 0)
        nx = 0
        for x in xs:
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
            
            result = tuple((r + y) for (r, y) in zip(result, (lw1, lb1, lw2, lb2, lw3, lb3)))
            nx += 1
        
        result = tuple(r / nx for r in result)
        return result
        

    def fiterate(self, datapoints, rate):
        n = 0
        lw1, lw2, lw3, lb1, lb2, lb3 = 0, 0, 0, 0, 0, 0
        for (x, y) in datapoints:
            z1 = self.W_1.dot(x) + self.b_1
            a1 = np.tanh(z1)
            z2 = self.W_2.dot(a1) + self.b_2
            a2 = np.tanh(z2)
            z3 = self.W_3.dot(a2) + self.b_3
            a3 = np.tanh(z3)

            delta3 = ((a3 - y) * (1 - np.tanh(z3) * np.tanh(z3))).transpose()
            delta2 = delta3 @ self.W_3 * (1 - np.tanh(z2) * np.tanh(z2)).transpose()
            delta1 = delta2.dot(self.W_2) * (1 - np.tanh(z1) * np.tanh(z1)).transpose()

            lw1 += delta1.transpose() @ x
            lw2 += delta2.transpose() @ a1.transpose()
            lw3 += delta3.transpose() @ a2.transpose()
            lb1 += delta1
            lb2 += delta2
            lb3 += delta3

            n += 1
        lw1 /= n
        lw2 /= n
        lw3 /= n
        lb1 /= n
        lb2 /= n
        lb3 /= n

        self.W_1 -= lw1 * rate
        self.b_1 -= (lb1 * rate).transpose()
        self.W_2 -= lw2 * rate
        self.b_2 -= (lb2 * rate).transpose()
        self.W_3 -= lw3 * rate
        self.b_3 -= (lb3 * rate).transpose()
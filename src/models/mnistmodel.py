import numpy as np

class MnistModel():
    def __init__(self):
        self.rng = np.random.default_rng(seed=43)
        
        # Parameter variables
        # Layer 1
        self.W1 = self.rng.normal(0, np.sqrt(2 / 784), size=(256, 784))
        self.b1 = self.rng.normal(0, np.sqrt(2 / 256), size=(256, 1))
        # Layer 2
        self.W2 = self.rng.normal(0, np.sqrt(2 / 256), size=(128, 256))
        self.b2 = self.rng.normal(0, np.sqrt(2 / 128), size=(128, 1))
        # Layer 3
        self.W3 = self.rng.normal(0, np.sqrt(2 / 128), size=(10, 128))
        self.b3 = self.rng.normal(0, np.sqrt(2 / 10), size=(10, 1))

        # Inference cache - at all times, these state variables contain the values
        # computed during the last inference pass
        self.z1 = np.zeros((1, 256))
        self.a1 = np.zeros((1, 256))
        self.z2 = np.zeros((1, 128))
        self.a2 = np.zeros((1, 128))
        self.z3 = np.zeros((1, 10))
        self.a3 = np.zeros((1, 10))


    # We use leaky ReLU as our activation function
    def __activation(self, v):
        return np.where(v < 0, 0.01 * v, v)


    # The jacobian of the activation function
    def __activation_jacobian(self, v):
        return np.diag(np.where(v < 0, 0.01, 1.0).flatten())
    

    # Update state variables to hold the results of performing inference on the given
    # vector
    def __infer(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.__activation(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.__activation(self.z2)
        self.z3 = self.W3 @ self.a2 + self.b3
        self.a3 = self.__activation(self.z3)

    
    # A function that returns the result of inference on the given variable
    def inference(self, x):
        self.__infer(x)
        return self.a3
    
    
    def loss(self, x, y):
        d = y - self.inference(x)
        return 0.5 * (d.transpose() @ d)


    def loss_gradient(self, x, y):
        d = self.inference(x) - y
        # TODO rederive using chain rule
        delta3 = self.__activation_jacobian(self.z3) @ d
        delta2 =  self.__activation_jacobian(self.z2) @ self.W3.T @ delta3
        delta1 = self.__activation_jacobian(self.z1) @ self.W2.T @ delta2

        lw1 = delta1 @ x.T
        lb1 = delta1
        lw2 = delta2 @ self.a1.T
        lb2 = delta2
        lw3 = delta3 @ self.a2.T
        lb3 = delta3

        return (lw1, lb1, lw2, lb2, lw3, lb3)


    # Batch gradient descent on a given dataset, given as a list of (input vector, output vector) pairs.
    def fiterate(self, datapoints, rate):
        avg_dw1, avg_db1, avg_dw2, avg_db2, avg_dw3, avg_db3 = (0, 0, 0, 0, 0, 0)
        n = 0
        for pair in datapoints:
            x = pair[0]
            y = pair[1]
            dw1, db1, dw2, db2, dw3, db3 = self.loss_gradient(x, y)
            avg_dw1 += dw1
            avg_db1 += db1
            avg_dw2 += dw2
            avg_db2 += db2
            avg_dw3 += dw3
            avg_db3 += db3

            n += 1

        avg_dw1 /= n
        avg_db1 /= n
        avg_dw2 /= n
        avg_db2 /= n
        avg_dw3 /= n
        avg_db3 /= n

        self.W1 -= avg_dw1 * rate
        self.b1 -= (avg_db1 * rate)
        self.W2 -= avg_dw2 * rate
        self.b2 -= (avg_db2 * rate)
        self.W3 -= avg_dw3 * rate
        self.b3 -= (avg_db3 * rate)
            

    # Iterate through the given dataset in batches, applying gradient descent to the model
    # on each batch
    def fit(self, dataset, batch_size=1):
        batches = self.batched(dataset, batch_size)
        for pairs in batches:
            self.fiterate(pairs, 0.001)


    def batched(self, collection, size):
        n = 0
        batch = []
        for x in collection:
            batch.append(x)
            n += 1
            if n == size:
                yield batch
                batch = []
                n = 0
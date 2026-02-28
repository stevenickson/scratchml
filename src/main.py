import numpy as np

import models.mini
import models.hand
import models.training

# Loads the MNIST datasets from the CSV files
# mnist_test.csv and mnist_train.csv directly into memory
# as a 2-dimensional array. This is not advisable on large
# workloads, but the MNIST dataset easily fits into
# memory on a modern computer and using the naive approach
# keeps the focus on machine learning rather than infrastructure
def load_datasets():
    training = np.loadtxt("mnist_train.csv.gz", delimiter=",")
    test = np.loadtxt("mnist_test.csv.gz", delimiter=",")
    return (training, test)


def main():
    training_set, test_set = load_datasets()
    m1 = models.mini.MiniModel()

    def batch_loss():
        avg = m1.loss([[1], [-2], [0.5]])
        for i in range(-50, 50, 1):
            for j in range(-50, 50, 1):
                avg += m1.loss([[1], [i / 10], [j / 10]])
        avg /= 10000
        return avg

    def batch_loss_gradient():
        avg = np.zeros((5))
        for i in range(-10, 10, 1):
            for j in range(-10, 10, 1):
                avg += m1.loss_gradient([[1], [i], [j]])
        avg /= 100
        return avg

    print(m1.inference([[1], [-5.5], [4]]))
    delta = 0.05
    for i in range(1, 10):
        m1.set_parameters(m1.get_parameters() - delta * batch_loss_gradient())
        delta = delta * 0.8
    print(m1.inference([[1], [-5.5], [4]]))

def scratch():
    m = models.hand.HandModel()
    print(m.inference(np.array([[0.6]])))
    dataset = [(np.array([[a]]), np.array([[0.5 * a], [0.5 * a]])) for a in np.arange(0.0, 1.0, 0.1)]
    for i in range(1, 10000):
        m.fiterate(dataset, 0.01)
    print(m.inference(np.array([[0.25]])))

if __name__ == "__main__":
    scratch()

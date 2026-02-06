import numpy as np


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


if __name__ == "__main__":
    main()

import numpy as np

import models.hand
import models.mnistmodel

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


# The model we created outputs a column vector of probabilities, representing the
# degree of confidence the model has in each possible answer. However,
# our dataset gives its training labels as floats - each image array
# is associated with the number it represents. To use this with our model,
# we need to convert each digit (given as a float) into the format that
# we expect our model to output. Thus, the given the float 5.0, we should output
# [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]. This function makes that
# conversion.
def float_digit_to_probability_vector(y):
    a = np.repeat(0.0, 10)[..., None]
    
    if y not in range(0, 10):
        return a
    i = int(y)
    a[i][0] = 1.0
    return a


def make_pairs():
    None


def most_confident(probability_vector):
    m = max(probability_vector)
    return probability_vector.tolist().index(m)
    
    
def mnist_model_demonstration():
    training_data, test_data = load_datasets()
    
    training_pairs = list(map(lambda a: (np.array(a[1:])[..., None] / 255, float_digit_to_probability_vector(a[0])), training_data))
    test_pairs = list(map(lambda a: (np.array(a[1:])[..., None] / 255, float_digit_to_probability_vector(a[0])), test_data))
    
    m1 = models.mnistmodel.MnistModel()
    m1.fit(training_pairs, batch_size=20)
    m1.fit(training_pairs, batch_size=15)
    m1.fit(training_pairs, batch_size=16)
    
    print(most_confident(m1.inference(test_pairs[0][0])), most_confident(test_pairs[0][1]))
    print(most_confident(m1.inference(test_pairs[1][0])), most_confident(test_pairs[1][1]))
    print(most_confident(m1.inference(test_pairs[2][0])), most_confident(test_pairs[2][1]))
    print(most_confident(m1.inference(test_pairs[3][0])), most_confident(test_pairs[3][1]))
    print(most_confident(m1.inference(test_pairs[4][0])), most_confident(test_pairs[4][1]))
    print(most_confident(m1.inference(test_pairs[5][0])), most_confident(test_pairs[5][1]))
    print(most_confident(m1.inference(test_pairs[6][0])), most_confident(test_pairs[6][1]))


def hand_model_demonstration():
    # Train the hand model to divide its input in half. Input is a 1x1 matrix [[x]], output is
    # a vector which should have the form <x, 0.5x>
    m = models.hand.HandModel()
    print(m.inference(np.array([[0.3]])))
    dataset = [(np.array([[a]]), np.array([[a], [0.5 * a]])) for a in np.arange(0.0, 1.0, 0.1)]
    for i in range(1, 50000):
        m.fiterate(dataset, 0.01)
    print(m.inference(np.array([[0.75]])))


def main():
    # hand_model_demonstration()
    mnist_model_demonstration()
    
    
if __name__ == "__main__":
    main()

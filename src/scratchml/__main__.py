import numpy as np

import scratchml.dataloader as dl
from scratchml.initializers import HeNormal
from scratchml.layers import MSE, Constant, Product, ReLU, Sum, Tanh
from scratchml.optimizers import SGD


def graphed_demonstration():
    x = np.array([[5.0, 10.0, 0.1]])
    c = np.array(
            [[ 0.42, -1.13,  0.77],
             [ 0.05,  0.91, -0.38],
             [-0.64,  0.28,  1.20]]
        )

    X = Constant(x)
    C = Constant(c)
    Y = Product(X, Product(C, C))
    Y()
    Y.backpropagate()



    print("=== X: ===")
    print(X.evaluation)
    print("=== C: ===")
    print(C.evaluation)
    print("=== Y: ===")
    print(Y.evaluation)

    print()

    print("=== dYdX: ===")
    print(X.backpropagation)
    print("=== dYdC: ===")
    print(C.backpropagation)

    print()

    print("=== Finite difference check ===")
    epsilon = np.array([0.001, 0.0, 0.0])
    X.evaluation = x + epsilon
    Y()
    high = Y.evaluation
    X.evaluation = x - epsilon
    Y()
    low = Y.evaluation
    print(high - low)


def perceptron_demonstration():
    X_train = [
                np.random.uniform(size=(1, 3), low=-1.0, high=1.0) for _ in range(1, 1000)
              ]
    Y_train = [x**2 for x in X_train]


    X = Constant(X_train[0])
    Y = Constant(Y_train[0])

    W_1 = Constant(np.random.rand(3, 32))
    b_1 = Constant(np.random.rand(1, 32))
    l1 = Tanh(Sum(Product(X, W_1), b_1))

    W_2 = Constant(np.random.rand(32, 3))
    b_2 = Constant(np.random.rand(1, 3))
    l2 = Tanh(Sum(Product(l1, W_2), b_2))

    """
    error = Sum(
                Product(
                    Y,
                    Constant(-1.0 * np.identity(3)),
                ),
                l2
            )

    # Mean squared error
    loss = Product(
              Constant(np.array([[1 / 3]])),
              Product(
                  error,
                  Transpose(error)
              )
           )
    """
    loss = MSE(l2, Y)

    avg = 0
    for x, y in zip(X_train, Y_train):
        X.evaluation = x
        Y.evaluation = y
        loss()
        avg += loss.evaluation.item()
    avg /= len(X_train)
    print(avg)

    opt = SGD(loss, parameters=[W_1, W_2, b_1, b_2], rate=0.15 / len(X_train), beta=0.9)
    for _ in range(1, 4000):
        for x, y in zip(X_train, Y_train):
            X.evaluation = x
            Y.evaluation = y
            loss()
            loss.backpropagate()
            opt.step()

    """
    gW_1 = np.zeros_like(W_1.evaluation)
    gb_1 = np.zeros_like(b_1.evaluation)
    gW_2 = np.zeros_like(W_2.evaluation)
    gb_2 = np.zeros_like(b_2.evaluation)
    beta = 0.9
    lr = 0.15
    for _ in range(1, 4500):
        gW_1 *= beta
        gb_1 *= beta
        gW_2 *= beta
        gb_2 *= beta

        meanloss = 0
        for x, y in zip(X_train, Y_train):
            X.evaluation = x
            Y.evaluation = y
            loss()
            meanloss += loss.evaluation / len(X_train)
            loss.backpropagate()
            gW_1 += W_1.backpropagation * (1 / len(X_train))
            gb_1 += b_1.backpropagation * (1 / len(X_train))
            gW_2 += W_2.backpropagation * (1 / len(X_train))
            gb_2 += b_2.backpropagation * (1 / len(X_train))
        print("Loss:", meanloss)

        W_1.evaluation -= gW_1 * lr
        b_1.evaluation -= gb_1 * lr
        W_2.evaluation -= gW_2 * lr
        b_2.evaluation -= gb_2 * lr
    """

    avg = 0
    for x, y in zip(X_train, Y_train):
        X.evaluation = x
        Y.evaluation = y
        loss()
        avg += loss.evaluation.item()
    avg /= len(X_train)
    print(avg)

    X_test = [
                np.random.uniform(size=(1, 3), low=-1.0, high=1.0) for _ in range(1, 1000)
              ]
    Y_test = [x**2 for x in X_test]
    avg = 0
    for x, y in zip(X_test, Y_test):
        X.evaluation = x
        Y.evaluation = y
        loss()
        avg += loss.evaluation.item()
    avg /= len(X_test)
    print(avg)


def perceptron_demonstration_batched():
    generator = np.random.default_rng(seed=312452)

    X_train = generator.uniform(size=(1000, 3), low=-1.0, high=1.0)
    X_test = generator.uniform(size=(1000, 3), low=-1.0, high=1.0)
    Y_train = X_train**2
    Y_test = X_test**2

    initializer = HeNormal(generator)

    X = Constant(X_train)
    Y = Constant(Y_train)

    W_1 = Constant(initializer((3, 32)))
    b_1 = Constant(np.random.rand(1, 32))
    l1 = ReLU(Sum(Product(X, W_1), b_1), alpha=0.1)

    W_2 = Constant(initializer((32, 3)))
    b_2 = Constant(np.random.rand(1, 3))
    l2 = ReLU(Sum(Product(l1, W_2), b_2), alpha=0.1)

    loss = MSE(l2, Y)

    loss()
    print(loss.evaluation)

    opt = SGD(loss, parameters=[W_1, W_2, b_1, b_2], rate=0.1, beta=0.9)
    for _ in range(10000):
        loss()
        loss.backpropagate()
        opt.step()

    loss()
    print(loss.evaluation)
    X.evaluation = X_test
    Y.evaluation = Y_test
    loss()
    print(loss.evaluation)


def mnist_demonstration():
    generator = np.random.default_rng(seed=312452)

    X_train, Y_train, X_test, Y_test = dl.load_fashion_mnist(flatten=True, normalize=True, vectorize_output=True)

    initializer = HeNormal(generator)

    X = Constant(X_train)
    Y = Constant(Y_train)

    W_1 = Constant(initializer((784, 256)))
    b_1 = Constant(np.random.rand(1, 256))
    l1 = ReLU(Sum(Product(X, W_1), b_1), alpha=0.1)

    W_2 = Constant(initializer((256, 128)))
    b_2 = Constant(np.random.rand(1, 128))
    l2 = ReLU(Sum(Product(l1, W_2), b_2), alpha=0.1)

    W_3 = Constant(initializer((128, 10)))
    b_3 = Constant(np.random.rand(1, 10))
    l3 = ReLU(Sum(Product(l2, W_3), b_3), alpha=0.1)

    loss = MSE(l3, Y)

    X.evaluation = X_test
    Y.evaluation = Y_test
    loss()
    print("Initial test loss:", loss.evaluation)
    print("Initial test accuracy:",
          np.mean(np.argmax(l3.evaluation, axis=1) == np.argmax(Y_test, axis=1)))

    X.evaluation = X_train
    Y.evaluation = Y_train
    loss()
    print("Initial training loss:", loss.evaluation)
    print("Initial training accuracy:",
          np.mean(np.argmax(l3.evaluation, axis=1) == np.argmax(Y_train, axis=1)))


    opt = SGD(loss, parameters=[W_1, W_2, W_3, b_1, b_2, b_3], rate=0.05, beta=0.9)
    for _ in range(500):
        loss()
        loss.backpropagate()
        opt.step()

    X.evaluation = X_test
    Y.evaluation = Y_test
    loss()
    print("Final test loss:", loss.evaluation)
    print("Final test accuracy:",
          np.mean(np.argmax(l3.evaluation, axis=1) == np.argmax(Y_test, axis=1)))

    X.evaluation = X_train
    Y.evaluation = Y_train
    loss()
    print("Final training loss:", loss.evaluation)
    print("Final training accuracy:",
          np.mean(np.argmax(l3.evaluation, axis=1) == np.argmax(Y_train, axis=1)))

def main():
    # hand_model_demonstration()
    # mnist_model_demonstration()
    # sequential_demonstration()
    # graphed_demonstration()
    # perceptron_demonstration()
    # perceptron_demonstration_batched()
    mnist_demonstration()


if __name__ == "__main__":
    main()

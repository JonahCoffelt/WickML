from NeuralNetworks import NeuralNetwork, Linear, ReLU, CrossEntropy
from Helpers import load_mnist, train_valid_test_split, accuracy
from Optimizers import SGD


def test_regularization():
    # Load data
    print("Loading data...")
    X, Y = load_mnist()
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train_valid_test_split(X, Y)
    X_train, Y_train = X_train[:2000], Y_train[:2000]

    # Create model
    print("Creating an mnist model...")
    model_none = NeuralNetwork(
        (
            Linear(784, 128, 0), 
            ReLU(), 
            Linear(128, 10, 0)
        ),
        CrossEntropy()
    )

    model_regularization = NeuralNetwork(
        (
            Linear(784, 128, 3e-3), 
            ReLU(), 
            Linear(128, 10, 3e-3)
        ),
        CrossEntropy()
    )

    mu, sigma = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / sigma
    X_valid = (X_valid - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Train
    print("Begin training no regularization:")
    SGD(0.1, 64, 100).train(model_none, X_train, Y_train)
    print("Begin training with regularization:")
    SGD(0.1, 64, 100).train(model_regularization, X_train, Y_train)

    # Test
    print("Begin testing:")
    print("MNIST test accuracy (no regularization):", accuracy(model_none, X_test, Y_test))
    print("MNIST test accuracy (with regularization):", accuracy(model_regularization, X_test, Y_test))


if __name__ == "__main__":
    test_regularization()
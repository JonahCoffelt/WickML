from NeuralNetworks import NeuralNetwork
from Helpers import np
import numpy as _np


class Optimizer():
    def __init__(self, learning_rate: float=0.1, batch_size: int=128, epochs: int=15, augment=None) -> None:
        """
        Train the neural network.
        Args:
            learning_rate: float: learning rate
            batch_size: int: batch size
            epochs: int: number of epochs
            augment: optional callable applied to each training batch of inputs, e.g. Helpers.RandomFlipCrop().
        """

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.augment = augment

        self.rng = _np.random.default_rng(42)

        # The model this optimizer's state (e.g. momentum or Adam moments) belongs to
        self.model = None

    def train(self, model: NeuralNetwork, X: np.ndarray, Y: np.ndarray) -> list[float]:
        """
        Train the neural network.
        Args:
            X: numpy array (n_samples, in_dim) of input features
            Y: numpy array (n_samples, out_dim) of target features
        Returns:
            list of the mean training loss of each epoch
        """

        for layer in model.layers:
            layer.training = True

        # State persists across train() calls on the same model, so a learning rate schedule can chain calls
        if self.model is not model:
            self.model = model
            self.init_state()

        n_samples = len(X)
        history = []
        for epoch in range(self.epochs):
            order = self.rng.permutation(n_samples)
            loss_accumulation = 0.0

            for start in range(0, n_samples, self.batch_size):
                idx = order[start:start + self.batch_size]

                x = X[idx]
                y = Y[idx]
                if self.augment is not None:
                    x = self.augment(x)

                logits = model.forward(x)
                loss = model.loss(logits, y)
                model.backprop()
                self.step()

                loss_accumulation += loss * len(idx)

            history.append(float(loss_accumulation / n_samples))
            print(f"Epoch {epoch + 1} - Loss {history[-1]:.4f}")

        return history

    def step(self):
        raise NotImplementedError

    def init_state(self) -> None:
        pass
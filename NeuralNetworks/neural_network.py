from Helpers import np
from NeuralNetworks import *

class NeuralNetwork:
    def __init__(self, layers: tuple[Layer, ...], loss_layer: Loss) -> None:
        self.layers = layers
        self.loss_layer = loss_layer

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        Args:
            x: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of output features
        """
        
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass through the loss layer.
        Args:
            pred: numpy array (n_samples, out_dim) of predicted features
            target: numpy array (n_samples, out_dim) of target features
        Returns:
            float: loss
        """

        return self.loss_layer.forward(pred, target)

    def backprop(self) -> None:
        """
        Backward pass through the neural network.
        """

        sensitivity = self.loss_layer.backprop()
        for layer in reversed(self.layers):
            sensitivity = layer.backprop(sensitivity)

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Predict the class of the input data.
        Args:
            X: numpy array (n_samples, in_dim) of input features
            batch_size: int: number of samples per forward pass, to bound memory use (e.g. conv im2col buffers)
        Returns:
            numpy array (n_samples,) of predicted class indices
        """
        for layer in self.layers:
            layer.training = False

        return np.concatenate([
            np.argmax(self.forward(X[start:start + batch_size]), axis=1)
            for start in range(0, len(X), batch_size)
        ])
from NeuralNetworks.layer import Layer
from NeuralNetworks import softmax
# import numpy as np
from Helpers import np


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, weight_penalty: float=0.0) -> None:
        """
        Create a new Linear layer.
        Args:
            in_dim: number of input features
            out_dim: number of output features
        """
        super().__init__()
        self.rng = np.random.default_rng(42)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_penalty = weight_penalty

        # Initialize the weights and biases
        # self.W = self.rng.standard_normal((in_dim, out_dim)) * np.sqrt(2 / in_dim)
        # self.W = self.rng.random((in_dim, out_dim)) * 2 - 1
        self.W = (self.rng.standard_normal((in_dim, out_dim)) * np.sqrt(2 / (in_dim + out_dim))).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        
        # Initalize the input and gradient states
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = np.zeros(in_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Args:
            x: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of output features
        """
        self.x = x
        return x @ self.W + self.b

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of this layers weights and bias.
        Args:
            sensitivity: numpy array (n_samples, out_dim) of gradient of the loss function with respect to the output of this layer
        Returns:
            numpy array (n_samples, in_dim) of gradient of the loss function with respect to the input of this layer
        """
        
        x_2d = self.x.reshape(-1, self.in_dim)
        s_2d = sensitivity.reshape(-1, self.out_dim)

        self.dW = x_2d.T @ s_2d + self.weight_penalty * self.W
        self.db = s_2d.sum(axis=0)
        return sensitivity @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]
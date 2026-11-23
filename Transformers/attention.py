from NeuralNetworks.layer import Layer
from Helpers import np

class Attention(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """TODO"""

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """TODO"""
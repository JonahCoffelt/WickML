from .optimizer import Optimizer
from NeuralNetworks import NeuralNetwork
from Helpers import np


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.1, batch_size: int = 128, epochs: int = 15, augment=None) -> None:
        super().__init__(learning_rate, batch_size, epochs, augment)

    def step(self) -> None:
        if self.model == None:
            raise ValueError
        
        for layer in self.model.layers:
            for param, grad in zip(layer.params(), layer.grads()):
                param -= grad * self.learning_rate
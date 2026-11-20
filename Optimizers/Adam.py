from .optimizer import Optimizer
from NeuralNetworks import NeuralNetwork
from Helpers import np


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, batch_size: int = 128, epochs: int = 15, augment=None,
                 beta1: float = 0.9, beta2: float = 0.999) -> None:
        """
        Args:
            beta1: decay rate of the running mean of the gradients (first moment)
            beta2: decay rate of the running mean of the squared gradients (second moment)
        Source:
            https://arxiv.org/pdf/1412.6980
        """
        super().__init__(learning_rate, batch_size, epochs, augment)
        self.beta1 = beta1
        self.beta2 = beta2

        # Per-parameter state, created by init_state() once a model is attached
        self.m = []
        self.v = []
        self.t = 0

    def init_state(self) -> None:
        """
        Zero the moment estimates and step count.
        """

        if self.model == None:
            raise ValueError

        params = [param for layer in self.model.layers for param in layer.params()]
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]
        self.t = 0

    def train(self, model: NeuralNetwork, X: np.ndarray, Y: np.ndarray) -> list[float]:
        # Clear vars

        return super().train(model, X, Y)

    def step(self) -> None:
        if self.model == None:
            raise ValueError
        
        # Increment time step
        self.t = self.t + 1

        params = [param for layer in self.model.layers for param in layer.params()]
        grads  = [grad  for layer in self.model.layers for grad  in layer.grads() ]
        epsilon = 1e-8

        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            # Bias correction
            m = self.m[i] / (1 - self.beta1 ** self.t)
            v = self.v[i] / (1 - self.beta2 ** self.t)

            # Update grad
            param -= self.learning_rate * m / (np.sqrt(v) + epsilon)
from .Adam import Adam


class AdamW(Adam):
    def __init__(self, learning_rate: float = 0.001, batch_size: int = 128, epochs: int = 15, augment=None,
                 beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01) -> None:
        """
        Adam with decoupled weight decay. The decay shrinks the weights directly instead of being added to the
        gradient, so it never passes through Adam's moment estimates (which would rescale it per parameter).
        Args:
            beta1: decay rate of the running mean of the gradients (first moment)
            beta2: decay rate of the running mean of the squared gradients (second moment)
            weight_decay: fraction of each weight removed per step, scaled by the learning rate
        Source:
            https://arxiv.org/pdf/1711.05101
        """
        super().__init__(learning_rate, batch_size, epochs, augment, beta1, beta2)
        self.weight_decay = weight_decay

    def step(self) -> None:
        if self.model == None:
            raise ValueError

        # Weight decay adjustment
        for layer in self.model.layers:
            for param in layer.params():
                if param.ndim > 1:
                    param -= self.learning_rate * self.weight_decay * param

        # Regular Adam update
        super().step()

from typing import Sequence
from NeuralNetworks.layer import Layer
from NeuralNetworks import Linear, Dropout, softmax
from Helpers import np

class Attention(Layer):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float=0.5) -> None:
        super().__init__()
        
        # Q, K, V projections
        self.Q_projection = Linear(embed_dim, embed_dim)
        self.K_projection = Linear(embed_dim, embed_dim)
        self.V_projection = Linear(embed_dim, embed_dim)
        self.Y_projection = Linear(embed_dim, embed_dim)

        # Dropout
        self.attention_dropout = Dropout(dropout)
        self.residual_dropout  = Dropout(dropout)

        self.n_heads = n_heads

    def forward(self, x: np.ndarray, is_causal: bool=True) -> np.ndarray:
        """
        Forward pass through the multi-head self-attention layer.
        Args:
            x: numpy array (batch_size, sequence_length, channels) of input token embeddings
            is_causal: if True, apply a causal mask so each position only attends to itself and earlier positions
        Returns:
            numpy array (batch_size, sequence_length, channels) of attended output features
        """

        # Get input dims
        batch_size, sequence_length, channels = x.shape

        # Project the Q, K, V from input
        Q = self.Q_projection(x)
        K = self.K_projection(x)
        V = self.V_projection(x)

        # Split into heads
        n_head_channels = channels // self.n_heads
        Q_heads = Q.reshape(batch_size, sequence_length, self.n_heads, n_head_channels).transpose(0, 2, 1, 3)
        K_heads = K.reshape(batch_size, sequence_length, self.n_heads, n_head_channels).transpose(0, 2, 1, 3)
        V_heads = V.reshape(batch_size, sequence_length, self.n_heads, n_head_channels).transpose(0, 2, 1, 3)

        # Save for back prop
        if self.training:
            self.Q_heads, self.K_heads, self.V_heads = Q_heads, K_heads, V_heads
            self.scale = 1.0 / np.sqrt(n_head_channels)


        # Attention
        # Compute QK and scale
        QK = Q_heads @ K_heads.swapaxes(-2, -1) / np.sqrt(n_head_channels)

        # Apply causal mask if specified
        if is_causal:
            mask = np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)
            QK = np.where(mask, -np.inf, QK)

        # Softmax
        attention = softmax(QK)
        if self.training: self.attention = attention  # Save for back prop

        # Dropout
        attention = self.attention_dropout(attention)
        if self.training: self.attention_dropped = attention  # Save for back prop

        # Attend to V
        y = attention @ V_heads

        # Concatenate heads
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, channels)

        # Y projection
        y = self.Y_projection(y)

        # Dropout
        y = self.residual_dropout(y)

        return y


    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Computes the gradients of the Q, K, V, and output projection weights and biases.
        Args:
            sensitivity: numpy array (batch_size, sequence_length, channels) of gradient of the loss function with respect to the output of this layer
        Returns:
            numpy array (batch_size, sequence_length, channels) of gradient of the loss function with respect to the input of this layer
        """
            
        batch_size, n_heads, sequence_length, n_head_channels = self.Q_heads.shape
        channels = n_heads * n_head_channels

        # Output dropout and proj
        dy = self.residual_dropout.backprop(sensitivity)
        dy = self.Y_projection.backprop(dy)

        # Un-merge heads
        dy = dy.reshape(batch_size, sequence_length, n_heads, n_head_channels).transpose(0, 2, 1, 3)

        # y = attention @ V
        d_attention = dy @ self.V_heads.swapaxes(-2, -1)
        dV_heads = self.attention_dropped.swapaxes(-2, -1) @ dy

        # Attention dropout
        d_attention = self.attention_dropout.backprop(d_attention)

        # Softmax
        attention = self.attention
        dQK = attention * (d_attention - np.sum(d_attention * attention, axis=-1, keepdims=True))

        # Scaling 
        dQK = dQK * self.scale

        # QK = Q @ K^T
        dQ_heads = dQK @ self.K_heads
        dK_heads = dQK.swapaxes(-2, -1) @ self.Q_heads

        # Merge heads back
        def merge_heads(t: np.ndarray) -> np.ndarray:
            return t.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, channels)

        # Sum input gradients
        return (self.Q_projection.backprop(merge_heads(dQ_heads))
            + self.K_projection.backprop(merge_heads(dK_heads))
            + self.V_projection.backprop(merge_heads(dV_heads)))


    def params(self): 
        return [p for l in (self.Q_projection, self.K_projection, self.V_projection, self.Y_projection) for p in l.params()]
   
    def grads(self):  
        return [g for l in (self.Q_projection, self.K_projection, self.V_projection, self.Y_projection) for g in l.grads()]


    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        self._training = value
        
        self.Q_projection.training = value
        self.K_projection.training = value
        self.V_projection.training = value
        self.Y_projection.training = value

        self.attention_dropout.training = value
        self.residual_dropout.training  = value


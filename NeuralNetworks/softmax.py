from Helpers import np

def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v, axis=-1, keepdims=True)  # Normalize for saftey (this is pretty standard)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)

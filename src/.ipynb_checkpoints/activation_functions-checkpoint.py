import cupy as np
from src.configuration import configuration

config = configuration.hyperParameter['activation']


def relu(x: np.array, dy: bool = False) -> np.array:
    if dy:
        return np.where(x > 0, 1, 0)

    return np.where(x > 0, x, 0)


def leakyRelu(x: np.array, scale: float = None, dy: bool = False) -> np.array:
    scale = config['leaky_relu']['scale'] if scale is None else scale
    if dy:
        return np.where(x > x * scale, 1, scale)

    return np.where(x > x * scale, x, x * scale)


def elu(x: np.array, scale: float = None, dy: bool = False) -> np.array:
    scale = config['elu']['scale'] if scale is None else scale
    if dy:
        return np.where(x >= 0, 1, scale * np.exp(x))

    return np.where(x >= 0, x, scale * (np.exp(x) - 1))


def sigmoid(x, dy=False):
    if dy:
        return sigmoid(x) * (1 - sigmoid(x))
    return (1 + np.exp(-x)) ** (-1)


def tanh(x, dy=False):
    if dy:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


def sin_x(x, dy=False):
    if dy:
        return np.cos(x)
    return np.sin(x)


def identity(x, dy=False):
    if dy:
        return np.ones_like(x)
    return x


def softMax(x: np.array, dy: bool = False) -> np.array:
    padding = configuration.hyperParameter['zero_division_padding']
    if dy:
        exps = np.exp(x - x.max())
        return exps / np.sum(exps)

    denominator = np.tile(np.sum(np.exp(x), axis=1), reps=(x.shape[1], 1)).transpose()
    return np.exp(x) / (denominator + padding)


__ACTIVATION_FUNCTIONS__ = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'identity': identity,
    'sin_x': sin_x,
    'softmax': softMax,
    'relu': relu,
    'elu': elu,
    'leaky_relu': leakyRelu,
}

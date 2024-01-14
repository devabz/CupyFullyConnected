from src.engine import kernel
from src.configuration import configuration

config = configuration.hyperParameter['activation']


def relu(x: kernel.kernel.ndarray, dy: bool = False) -> kernel.kernel.ndarray:
    if dy:
        return kernel.kernel.where(x > 0, 1, 0)

    return kernel.kernel.where(x > 0, x, 0)


def leakyRelu(x: kernel.kernel.ndarray, scale: float = None, dy: bool = False) -> kernel.kernel.ndarray:
    scale = config['leaky_relu']['scale'] if scale is None else scale
    if dy:
        return kernel.kernel.where(x > x * scale, 1, scale)

    return kernel.kernel.where(x > x * scale, x, x * scale)


def elu(x: kernel.kernel.ndarray, scale: float = None, dy: bool = False) -> kernel.kernel.ndarray:
    scale = config['elu']['scale'] if scale is None else scale
    if dy:
        return kernel.kernel.where(x >= 0, 1, scale * kernel.kernel.exp(x))

    return kernel.kernel.where(x >= 0, x, scale * (kernel.kernel.exp(x) - 1))


def sigmoid(x, dy=False):
    if dy:
        return sigmoid(x) * (1 - sigmoid(x))
    return (1 + kernel.kernel.exp(-x)) ** (-1)


def tanh(x, dy=False):
    if dy:
        return 1 - kernel.kernel.tanh(x) ** 2
    return kernel.kernel.tanh(x)


def sin_x(x, dy=False):
    if dy:
        return kernel.kernel.cos(x)
    return kernel.kernel.sin(x)


def identity(x, dy=False):
    if dy:
        return kernel.kernel.ones_like(x)
    return x


def softMax(x: kernel.kernel.ndarray, dy: bool = False) -> kernel.kernel.ndarray:
    padding = configuration.hyperParameter['zero_division_padding']
    if dy:
        exps = kernel.kernel.exp(x - x.max())
        return exps / kernel.kernel.sum(exps)

    denominator = kernel.kernel.tile(kernel.kernel.sum(kernel.kernel.exp(x), axis=1), reps=(x.shape[1], 1)).transpose()
    denominator = kernel.kernel.sum(kernel.kernel.exp(x))
    return kernel.kernel.exp(x) / (denominator + padding)


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

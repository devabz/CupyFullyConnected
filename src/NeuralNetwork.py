import numpy
import cupy
from src.activation_functions import __ACTIVATION_FUNCTIONS__
from src.optimizer import Optimizer
from src.configuration import configuration
from tqdm import tqdm
from src.loss_functions import __LossFunctionBase__
from src.engine import kernel
config = configuration.hyperParameter['activation']


class CudaLayer:
    def __init__(self, output_size, act_f=None, w_init=None, optimizer='sgd', lr=None):
        self.weights: kernel.kernel.ndarray = None
        self.bias: kernel.kernel.ndarray = None
        self.optimizer = None
        self.logs: dict = {}
        self.error = []
        self.history: dict = {
            'w': [],
            'g': [],
            'w_init': None
        }
        
        self.__output_size__ = output_size
        self.__optimizer_str__ = optimizer
        self.__activation_str__ = 'sigmoid' if act_f is None else act_f
        self.__learning_rate__ = configuration.hyperParameter['learning_rate'] if lr is None else lr
        self.__weight_initializer__: dict = {} if w_init is None else w_init

    def clear(self):
        self.history['w'] = []
        self.history['g'] = []
        self.logs = {}

    def initialize(self, arr, low=-1, high=1, mode=None):
        valid = ['normal', 'uniform', 'logistic', 'gamma', 'lognormal', 'binomial', 'gumbel']
        mode = 'uniform' if mode not in valid else mode
        sampler = getattr(kernel.kernel.random, mode)
        os = self.__output_size__

        if mode not in ['gamma', 'lognormal', 'binomial']:
            self.weights = sampler(low, high, size=(arr.shape[-1], os))
            self.bias = sampler(low, high, size=(1, os))
        elif mode == 'gamma':
            self.weights = sampler(high, size=(arr.shape[-1], os))
            self.bias = sampler(high, size=(1, os))
        elif mode == 'lognormal':
            self.weights = sampler(mean=low, sigma=high, size=(arr.shape[-1], os))
            self.bias = sampler(mean=low, sigma=high, size=(1, os))
        elif mode == 'binomial':
            self.weights = sampler(p=low, n=high, size=(arr.shape[-1], os))
            self.bias = sampler(p=low, n=high, size=(1, os))
        self.history['w_init'] = kernel.kernel.linalg.norm(self.weights).get()
        self.optimizer = Optimizer(
            shape=self.weights.shape,
            learningRate=self.__learning_rate__,
            option=self.__optimizer_str__
        )

    def activation(self, x, dy=False):
        return __ACTIVATION_FUNCTIONS__[self.__activation_str__](x=x, dy=dy)

    def forward(self, X):
        if self.weights is None:
            self.initialize(arr=X, **self.__weight_initializer__)

        z = kernel.kernel.matmul(X, self.weights) + self.bias
        act = self.activation(x=z)
        dact = self.activation(x=z, dy=True)
        self.logs = {'X': X, 'z': z, 'act': act, 'dact': dact}

        return act

    def backward(self, error, lr):
        self.history['w'].append(kernel.kernel.linalg.norm(self.weights).get())

        self.error.append(error)

        # Compute gradients
        backward = kernel.kernel.multiply(self.logs['dact'], error)
        new_weights = kernel.kernel.matmul(self.logs['X'].transpose(), backward)
        new_bias = kernel.kernel.mean(backward, axis=0)

        # Compute aux term for next layer
        backward = kernel.kernel.matmul(backward, self.weights.transpose())

        # Update the current layer
        self.weights -= lr * new_weights
        self.weights -= self.optimizer.optimizer.compute(new_weights)
        self.bias -= lr * new_bias
        return backward

    def set_lr(self, lr):
        self.__learning_rate__ = lr
        self.optimizer.optimizer.learningRate = lr

    def state(self):
        return [self.logs['act'].get(), self.weights.get()]

    def __repr__(self):
        return f'CudaLayer(shape={None, self.__output_size__}, act={self.__activation_str__}, opt={self.__optimizer_str__})'


class CudaNetwork:

    def __init__(self, poly_reg=False, round_off=30, __kernel__='cupy'):
        kernel.set_kernel(__kernel__)
        self.__round_off__ = round_off
        self.poly_reg: int = poly_reg
        self.layers: list[CudaLayer] = []
        self.__loss_function__: __LossFunctionBase__ = None

    def _single_fit(self, X, y, lr):
        predictions = self.predict(X=X)
        self.backward(self.__loss_function__.compute(X=predictions, y=y, dy=True), lr=lr)
        return kernel.kernel.sum(self.__loss_function__.compute(X=predictions, y=y, dy=False))

    def forward(self, X):
        if isinstance(X, numpy.ndarray) and kernel.__kernel_str__ == "cupy":
            X = kernel.kernel.asarray(X)
        elif isinstance(X, cupy.ndarray) and kernel.__kernel_str__ == "numpy":
            X = X.get()

        if self.poly_reg:
            X = kernel.kernel.power(X, kernel.kernel.arange(self.poly_reg))

        for layer in self.layers:
            X = layer.forward(X=X)
        return X

    def backward(self, error, lr):
        for layer in self.layers[::-1]:
            error = layer.backward(error, lr=lr)

    def fit(self, X, y, lr, epochs=10):
        if isinstance(y, numpy.ndarray):
            y = kernel.kernel.asarray(y)
            
        if self.__loss_function__ is None:
            raise AttributeError(f'No loss function set. Use Network().set_loss_function(__LossFunctionBase__)')
        losses = []
        for epoch in tqdm(range(epochs), desc="Training", leave=False, ncols=80):
            loss = self._single_fit(X=X, y=y, lr=lr)
            if kernel.__kernel_str__ == "cupy":
                losses.append(loss.get())
            else:
                losses.append(loss)

        return losses

    def predict(self, X):
        return kernel.kernel.round(self.forward(X=X), self.__round_off__)

    def add(self, layer: CudaLayer):
        if isinstance(layer, CudaLayer):
            self.layers.append(layer)
        else:
            raise TypeError(f'{type(layer)} is not {CudaLayer}')

    def describe(self):
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                print("output :", layer)
            else:
                print("hidden :", layer)

    def set_lr(self, lr):
        for layer in self.layers:
            layer.set_lr(lr=lr)

    def set_loss_function(self, loss_function: __LossFunctionBase__):
        if not isinstance(loss_function, __LossFunctionBase__):
            raise TypeError(f'{type(loss_function)} is not {__LossFunctionBase__}')

        self.__loss_function__ = loss_function

    def encode_weights(self):
        if kernel.__kernel_str__ == "cupy":
            return list([(layer.weights.get().flatten().tolist(), layer.weights.shape[-1]) for layer in self.layers])
        return list([(layer.weights.flatten().tolist(), layer.weights.shape[-1]) for layer in self.layers])

    def encode_bias(self):
        if kernel.__kernel_str__ == "cupy":
            return list([layer.bias.get().flatten().tolist() for layer in self.layers])
        return list([layer.bias.flatten().tolist() for layer in self.layers])

    def decode_weights(self, weights):
        for layer, weight in zip(self.layers, weights):
            layer.weights = kernel.kernel.array(weight[0]).reshape(-1, weight[-1])

    def decode_bias(self, bias):
        for layer, bias in zip(self.layers, bias):
            layer.bias = kernel.kernel.array(bias)

    def clear(self):
        for layer in self.layers:
            layer.clear()


if __name__ == '__main__':
    def experiment(mode):
        kernel.kernel.random.seed(0)
        l = CudaLayer(
            output_size=1,
            act_f='sigmoid',
            w_init={'low': -1, 'high': 1, 'mode': mode},
            optimizer='adaGrad',
            lr=1e-5,
        )

        x = kernel.kernel.arange(11)
        y = (x ** 2).reshape(-1, 1)
        x = kernel.kernel.vstack([kernel.kernel.zeros_like(x), x]).transpose()
        tmp = l.forward(X=x)
        return tmp

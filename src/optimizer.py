from src.engine import kernel
from src.configuration import configuration, validateOptions


class Optimizer:
    def __init__(self, option: str, shape: tuple, learningRate: float):
        self._sgd = self._SGD
        self._momentum = self._Momentum
        self._adaGrad = self._AdaGrad
        self._rmsProp = self._RMSProp
        self._adam = self._ADAM

        validateOptions(options=['sgd', 'momentum', 'adaGrad', 'rmsProp', 'adam'], option=option)

        self.optimizer = {
            'sgd': self._sgd,
            'momentum': self._momentum,
            'adaGrad': self._adaGrad,
            'rmsProp': self._rmsProp,
            'adam': self._adam
        }[option](shape=shape, learningRate=learningRate,
                  config=configuration.hyperParameter['optimizer'][option])

    class _Optimizer:
        def __init__(self, shape: tuple, learningRate: float, config: dict):
            self._config = config
            self.shape = shape
            self.learningRate = kernel.kernel.ones(shape=shape) * learningRate
            self._LR = learningRate

        def compute(self, gradients: kernel.kernel.array, **kwargs):
            pass

        def hyperparameter_tuning(self, **kwargs):
            pass

        def log(self):
            pass

        def reshape(self, shape):
            pass

    class _SGD(_Optimizer):
        def __init__(self, shape: tuple, learningRate: float, config: dict):
            super().__init__(shape=shape, learningRate=learningRate, config=config)
            pass

        def compute(self, gradients: kernel.kernel.array, **kwargs) -> kernel.kernel.array:
            result = kernel.kernel.multiply(gradients, self.learningRate)
            self.learning_rate_decay()
            return result

        def learning_rate_decay(self) -> None:
            self.learningRate *= self._config['learning_rate_decay']

        def reshape(self, shape):
            self.learningRate = kernel.kernel.ones(shape=shape) * self._LR

    class _Momentum(_Optimizer):
        def __init__(self, shape: tuple, learningRate: float, config: dict):
            super().__init__(shape=shape, learningRate=learningRate, config=config)
            self.momentum = self._config['initial_momentum'](*shape)
            self.beta = self._config['beta']

        def compute(self, gradients: kernel.kernel.array, **kwargs) -> kernel.kernel.array:
            self.momentum = self.compute_momentum(gradients=gradients)
            self.learning_rate_decay()
            return self.momentum

        def compute_momentum(self, gradients: kernel.kernel.array) -> kernel.kernel.array:
            return self.beta * self.momentum[-1] + kernel.kernel.multiply(self.learningRate, gradients)

        def learning_rate_decay(self) -> None:
            self.learningRate *= self._config['learning_rate_decay']

        def reshape(self, shape):
            self.learningRate = kernel.kernel.ones(shape=shape) * self._LR
            self.momentum = self._config['initial_momentum'](*shape)

    class _AdaGrad(_Optimizer):
        def __init__(self, shape: tuple, learningRate: float, config: dict):
            super().__init__(shape=shape, learningRate=learningRate, config=config)
            self.gradientMatrix = self._config['initial_gradient'](*shape)

        def compute(self, gradients: kernel.kernel.array, **kwargs) -> kernel.kernel.array:
            self.gradientMatrix = self.compute_gradient_matrix(gradients=gradients)
            epsilon = configuration.hyperParameter['zero_division_padding']
            result = kernel.kernel.multiply(self.learningRate, gradients) / (kernel.kernel.sqrt(self.gradientMatrix) + epsilon)

            return result

        def compute_gradient_matrix(self, gradients: kernel.kernel.array) -> kernel.kernel.array:
            self.gradientMatrix += kernel.kernel.multiply(gradients, gradients)
            return self.gradientMatrix

        def reshape(self, shape):
            self.learningRate = kernel.kernel.ones(shape=shape) * self._LR
            self.gradientMatrix = self._config['initial_gradient'](*shape)

        def learning_rate_decay(self) -> None:
            self.learningRate *= self._config['learning_rate_decay']
            return self.learningRate

    class _RMSProp(_Optimizer):
        def __init__(self, shape: tuple, learningRate: float, config: dict):
            super().__init__(shape=shape, learningRate=learningRate, config=config)
            self.gradientMatrix = self._config['initial_gradient'](*shape)
            self.beta = self._config['beta']

        def compute(self, gradients: kernel.kernel.array, **kwargs) -> kernel.kernel.array:
            self.gradientMatrix = self.compute_gradient_matrix(gradients=gradients)
            epsilon = configuration.hyperParameter['zero_division_padding']
            return kernel.kernel.multiply(self.learningRate, gradients) / (kernel.kernel.sqrt(self.gradientMatrix) + epsilon)

        def compute_gradient_matrix(self, gradients: kernel.kernel.array) -> kernel.kernel.array:
            self.gradientMatrix *= self.beta
            self.gradientMatrix += (1 - self.beta) * kernel.kernel.multiply(gradients, gradients)
            return self.gradientMatrix

        def reshape(self, shape):
            self.learningRate = kernel.kernel.ones(shape=shape) * self._LR
            self.gradientMatrix = self._config['initial_gradient'](*shape)

    class _ADAM(_Optimizer):
        def __init__(self, shape: tuple, learningRate: float, config: dict):
            super().__init__(shape=shape, learningRate=learningRate, config=config)
            self.gradientMatrix = self._config['initial_gradient'](*shape)
            self.momentumMatrix = self._config['initial_momentum'](*shape)
            self.momentumBeta = self._config['momentum_beta']
            self.gradientBeta = self._config['gradient_beta']

        def compute(self, gradients: kernel.kernel.array, **kwargs):
            epsilon = configuration.hyperParameter['zero_division_padding']
            self.gradientMatrix = self.compute_gradient_component(gradients=gradients)
            self.momentumMatrix = self.compute_momentum_component(gradients=gradients)
            return kernel.kernel.multiply(self.learningRate, self.momentumMatrix) / (kernel.kernel.sqrt(self.gradientMatrix) + epsilon)

        def compute_gradient_component(self, gradients: kernel.kernel.array) -> kernel.kernel.array:
            self.gradientMatrix *= self.gradientBeta
            self.gradientMatrix += (1 - self.gradientBeta) * kernel.kernel.multiply(gradients, gradients)
            self.gradientMatrix /= (1 - self.gradientBeta)
            return self.gradientMatrix

        def compute_momentum_component(self, gradients: kernel.kernel.array) -> kernel.kernel.array:
            self.momentumMatrix *= self.momentumBeta
            self.momentumMatrix += (1 - self.momentumBeta) * gradients
            self.momentumMatrix /= (1 - self.momentumBeta)
            return self.momentumMatrix

        def reshape(self, shape):
            self.learningRate = kernel.kernel.ones(shape=shape) * self._LR
            self.gradientMatrix = self._config['initial_gradient'](*shape)
            self.momentumMatrix = self._config['initial_momentum'](*shape)


if __name__ == '__main__':
    opt = Optimizer(shape=(10, 4), learningRate=configuration.hyperParameter['learning_rate'], option='sgd')
    print(opt.optimizer)
    print(opt.optimizer.learningRate)

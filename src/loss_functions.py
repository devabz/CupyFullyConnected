from src.engine import kernel
from abc import ABC


class __LossFunctionBase__(ABC):
    def compute(self, X, y, dy=False):
        pass


class MSE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        if dy:
            return X - y
        return kernel.kernel.mean((X - y) ** 2)


class MAE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return kernel.kernel.where(error > 0, 1, -1) / len(X)
        return kernel.kernel.mean(kernel.kernel.abs(error))


class MBE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return kernel.kernel.where(error > 0, 1, -1) / len(X)
        return kernel.kernel.mean(error)


class RAE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = kernel.kernel.abs(X - y)
        absolute_sum = kernel.kernel.sum(kernel.kernel.abs(y - kernel.kernel.mean(y)))
        if dy:
            return kernel.kernel.sign(X - y) / absolute_sum
        return kernel.kernel.sum(error) / absolute_sum


class RSE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        squared_error = kernel.kernel.sum(error ** 2)
        total_variation = kernel.kernel.sum((y - kernel.kernel.mean(y)) ** 2)
        if dy:
            return 2 * error / total_variation
        return squared_error / total_variation


class MAPE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = kernel.kernel.abs(X - y)
        percentage_error = error / kernel.kernel.abs(y)
        if dy:
            return kernel.kernel.sign(X - y) / kernel.kernel.abs(y)
        return kernel.kernel.mean(percentage_error) * 100


class RMSE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        squared_error = kernel.kernel.sum(error ** 2)
        mean_squared_error = squared_error / len(y)
        if dy:
            return 2 * error / len(y)
        return kernel.kernel.sqrt(mean_squared_error)


class MSLE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        log_X = kernel.kernel.log1p(X)
        log_y = kernel.kernel.log1p(y)
        squared_log_error = (log_X - log_y) ** 2
        if dy:
            return 2 * (log_X - log_y) / (1 + X)
        return kernel.kernel.mean(squared_log_error)


class HuberLoss(__LossFunctionBase__):
    def __init__(self, delta=1.0):
        self.delta = delta

    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return kernel.kernel.where(kernel.kernel.abs(error) <= self.delta, error,
                                       self.delta * kernel.kernel.sign(error))
        return kernel.kernel.where(kernel.kernel.abs(error) <= self.delta, 0.5 * error ** 2,
                                   self.delta * (kernel.kernel.abs(error) - 0.5 * self.delta))


class LogCoshLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return kernel.kernel.tanh(error)
        return kernel.kernel.mean(kernel.kernel.log(kernel.kernel.cosh(error)))


class QuantileLoss(__LossFunctionBase__):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return kernel.kernel.where(error >= 0, self.alpha, self.alpha - 1)
        return kernel.kernel.mean(kernel.kernel.where(error >= 0, self.alpha * error, (self.alpha - 1) * error))


class HingeLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        margin = 1 - X * y
        hinge_loss = kernel.kernel.maximum(0, margin)
        if dy:
            return kernel.kernel.where(margin > 0, -y, 0)
        return kernel.kernel.mean(hinge_loss)


class SigmoidCrossEntropyLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        epsilon = 1e-15
        X = kernel.kernel.clip(X, epsilon, 1 - epsilon)
        cross_entropy_loss = - (y * kernel.kernel.log(X) + (1 - y) * kernel.kernel.log(1 - X))
        if dy:
            return (X - y) / ((X - 1) * X + epsilon)
        return kernel.kernel.mean(cross_entropy_loss)


class WeightedCrossEntropyLoss(__LossFunctionBase__):
    def __init__(self, weights):
        self.weights = weights

    def compute(self, X, y, dy=False):
        epsilon = 1e-15
        X = kernel.kernel.clip(X, epsilon, 1 - epsilon)
        cross_entropy_loss = - (self.weights * (y * kernel.kernel.log(X) + (1 - y) * kernel.kernel.log(1 - X)))
        if dy:
            return self.weights * (X - y) / ((X - 1) * X + epsilon)
        return kernel.kernel.mean(cross_entropy_loss)


class SoftmaxCrossEntropyLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        epsilon = 1e-15
        exp_X = kernel.kernel.exp(X - kernel.kernel.max(X, axis=1, keepdims=True))
        softmax_probs = exp_X / kernel.kernel.sum(exp_X, axis=1, keepdims=True)
        cross_entropy_loss = - kernel.kernel.sum(
            y * kernel.kernel.log(kernel.kernel.clip(softmax_probs, epsilon, 1 - epsilon)), axis=1)
        if dy:
            return (softmax_probs - y) / y.shape[0]
        return kernel.kernel.mean(cross_entropy_loss)


class SparseCrossEntropyLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        epsilon = 1e-15
        exp_X = kernel.kernel.exp(X - kernel.kernel.max(X, axis=1, keepdims=True))
        softmax_probs = exp_X / kernel.kernel.sum(exp_X, axis=1, keepdims=True)
        cross_entropy_loss = - kernel.kernel.log(kernel.kernel.clip(softmax_probs[:, y], epsilon, 1 - epsilon))
        if dy:
            grad = softmax_probs.copy()
            grad[:, y] -= 1
            return grad / y.shape[0]
        return kernel.kernel.mean(cross_entropy_loss)


class KullbackLeiblerDivergenceLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        epsilon = 1e-15
        X = kernel.kernel.clip(X, epsilon, None)
        y = kernel.kernel.clip(y, epsilon, None)
        kl_divergence_loss = kernel.kernel.sum(y * kernel.kernel.log(y / X), axis=1)
        if dy:
            return -y / X / y.shape[0]
        return kernel.kernel.mean(kl_divergence_loss)


class BinaryClass:
    hinge: HingeLoss = HingeLoss()
    sigmoidCE: SigmoidCrossEntropyLoss = SigmoidCrossEntropyLoss()
    weightedCE: WeightedCrossEntropyLoss = lambda weights: WeightedCrossEntropyLoss(weights=weights)


class Multiclass:
    oftmaxCE: SoftmaxCrossEntropyLoss = SoftmaxCrossEntropyLoss()
    sparseCE: SparseCrossEntropyLoss = SparseCrossEntropyLoss()
    kl_divergence: KullbackLeiblerDivergenceLoss = KullbackLeiblerDivergenceLoss()


class ClassificationLoss:
    binary: BinaryClass = BinaryClass
    multiple: Multiclass = Multiclass


class RegressionLoss:
    mape: MAPE = MAPE()
    rmse: RMSE = RMSE()
    mse: MSE = MSE()
    rae: RAE = RAE()
    mbe: MBE = MBE()
    mae: MAE = MAE()
    rse: RSE = RSE()
    msle: MSLE = MSLE()
    huber: HuberLoss = HuberLoss()
    log_cosh: LogCoshLoss = LogCoshLoss()
    quantile: QuantileLoss = QuantileLoss()


class LossFunctions:
    regression = RegressionLoss
    classification = ClassificationLoss
    mape: MAPE = MAPE()
    rmse: RMSE = RMSE()
    mse: MSE = MSE()
    rae: RAE = RAE()
    mbe: MBE = MBE()
    mae: MAE = MAE()
    rse: RSE = RSE()
    msle: MSLE = MSLE()
    huber: HuberLoss = HuberLoss()
    log_cosh: LogCoshLoss = LogCoshLoss()
    quantile: QuantileLoss = QuantileLoss()
    hinge: HingeLoss = HingeLoss()
    sigmoidCE: SigmoidCrossEntropyLoss = SigmoidCrossEntropyLoss()
    weightedCE: WeightedCrossEntropyLoss = lambda weights: WeightedCrossEntropyLoss(weights=weights)
    softmaxCE: SoftmaxCrossEntropyLoss = SoftmaxCrossEntropyLoss()
    sparseCE: SparseCrossEntropyLoss = SparseCrossEntropyLoss()
    kl_divergence: KullbackLeiblerDivergenceLoss = KullbackLeiblerDivergenceLoss()

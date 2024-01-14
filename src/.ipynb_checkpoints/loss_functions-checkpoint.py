import cupy as np
from abc import ABC


class __LossFunctionBase__(ABC):
    def compute(self, X, y, dy=False):
        pass


class MSE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        if dy:
            return X - y
        return np.mean((X - y) ** 2)


class MAE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return np.where(error > 0, 1, -1) / len(X)
        return np.mean(np.abs(error))


class MBE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return np.where(error > 0, 1, -1) / len(X)
        return np.mean(error)


class RAE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = np.abs(X - y)
        absolute_sum = np.sum(np.abs(y - np.mean(y)))
        if dy:
            return np.sign(X - y) / absolute_sum
        return np.sum(error) / absolute_sum


class RSE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        squared_error = np.sum(error ** 2)
        total_variation = np.sum((y - np.mean(y)) ** 2)
        if dy:
            return 2 * error / total_variation
        return squared_error / total_variation


class MAPE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = np.abs(X - y)
        percentage_error = error / np.abs(y)
        if dy:
            return np.sign(X - y) / np.abs(y)
        return np.mean(percentage_error) * 100


class RMSE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        squared_error = np.sum(error ** 2)
        mean_squared_error = squared_error / len(y)
        if dy:
            return 2 * error / len(y)
        return np.sqrt(mean_squared_error)


class MSLE(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        log_X = np.log1p(X)
        log_y = np.log1p(y)
        squared_log_error = (log_X - log_y) ** 2
        if dy:
            return 2 * (log_X - log_y) / (1 + X)
        return np.mean(squared_log_error)


class HuberLoss(__LossFunctionBase__):
    def __init__(self, delta=1.0):
        self.delta = delta

    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return np.where(np.abs(error) <= self.delta, error, self.delta * np.sign(error))
        return np.where(np.abs(error) <= self.delta, 0.5 * error ** 2, self.delta * (np.abs(error) - 0.5 * self.delta))


class LogCoshLoss(__LossFunctionBase__):
    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return np.tanh(error)
        return np.mean(np.log(np.cosh(error)))


class QuantileLoss(__LossFunctionBase__):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def compute(self, X, y, dy=False):
        error = X - y
        if dy:
            return np.where(error >= 0, self.alpha, self.alpha - 1)
        return np.mean(np.where(error >= 0, self.alpha * error, (self.alpha - 1) * error))


class LossFunctions:
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


loss_functions = LossFunctions
import numpy as np
from numpy.linalg import inv


class LinearRegressor:
    """
        Simple multivariate linear regression using ordinary least square
    """
    coefs = None

    def fit(self, X: np.matrix, y: np.matrix):
        tp_X = np.transpose(X)
        self.coefs = inv(tp_X * X) * tp_X * np.transpose(y)

    def predict(self, target: np.matrix):
        return target * self.coefs


import collections

from typing import Optional

import numpy as np
import sklearn.base
import matplotlib.pyplot as plt

from ..misc import utils

ApproxAndParams = collections.namedtuple("ApproxAndParams", ["predicted",
                                                             "params"])

class BaseLocalModel(sklearn.base.TransformerMixin,
                     sklearn.base.BaseEstimator):
    def __init__(self, *args, **kwargs):
        self._X: Optional[np.ndarray] = None
        self._X_approx: Optional[np.ndarray] = None
        self._T: Optional[np.ndarray] = None
        self._params: Optional[np.ndarray] = None

    def set_time(self, time: np.ndarray) -> None:
        self._T = time.copy()

    def fit_row(self, row: np.ndarray) -> ApproxAndParams:
        raise NotImplementedError()

    def fit(self, X, y=None) -> 'BaseLocalModel':
        X = utils.to_matrix(X)
        self._X = X
        self._T = np.arange(0, X.shape[1])
        rows_and_params = [self.fit_row(X[i, :]) for i in range(X.shape[0])]
        self._X_approx = np.array([pair.predicted for pair in rows_and_params])
        self._params = np.array([pair.params for pair in rows_and_params])
        return self

    def transform(self, X=None):
        return self._params

    def plot(self, idx: int = 0) -> plt.Figure:
        if self._X is None or self._T is None:
            raise ValueError("Data is not provided. Nothing to plot")
        if self._X_approx is None:
            raise ValueError("X is not fitted yet")

        plt.plot(self._T, self._X[idx, :])
        plt.plot(self._T, self._X_approx[idx, :])
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.legend(["Original signal", "Approximated signal"])
        return plt.figure()

    def get_predicted(self) -> np.ndarray:
        if len(self._X_approx.shape) == 1:
            return self._X_approx.reshape((1, -1))
        return self._X_approx
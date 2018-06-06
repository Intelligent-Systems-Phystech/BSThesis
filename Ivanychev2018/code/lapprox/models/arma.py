"""ARMA local transformer.

Author: Sergey Ivanychev
"""
import numpy as np
import statsmodels.tsa.arima_model as arima_model

from . import base

class Arma(base.BaseLocalModel):
    def __init__(self, p: int, q: int) -> None:
        base.BaseLocalModel.__init__(self, p, q)
        self._p = p
        self._q = q

    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        model = arima_model.ARMA(row, (self._p, self._q)).fit()
        predicted = model.predict(start=0, end=len(row) - 1)
        params = model.params
        return base.ApproxAndParams(predicted, params)
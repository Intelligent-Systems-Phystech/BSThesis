"""Auto regression local transformer.

Author: Sergey Ivanychev
"""
import numpy as np
import statsmodels.tsa.ar_model as ar_model

from . import base



class Ar(base.BaseLocalModel):
    def __init__(self, degree: int) -> None:
        base.BaseLocalModel.__init__(self)
        self._degree = degree

    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        model = ar_model.AR(row).fit(maxlag=self._degree)
        head = row[:self._degree]
        tail = model.predict(start=self._degree, end=len(row) - 1)
        predicted = np.concatenate((head, tail))
        params = model.params
        return base.ApproxAndParams(predicted, params)
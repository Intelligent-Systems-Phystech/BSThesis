"""FFT local transformer.

Author: Sergey Ivanychev
"""
import numpy as np
import scipy.fftpack as fftpack

from . import base

class Fft(base.BaseLocalModel):
    def __init__(self, n_harmonics: int) -> None:
        base.BaseLocalModel.__init__(self, n_harmonics)
        self._n_harmonics = n_harmonics
        pass
    
    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        params = fftpack.fft(row)
        top_harmonics_idx = np.argsort(np.abs(params))[-self._n_harmonics:]
        predicted = fftpack.ifft(params)
        return base.ApproxAndParams(predicted, top_harmonics_idx[::-1])
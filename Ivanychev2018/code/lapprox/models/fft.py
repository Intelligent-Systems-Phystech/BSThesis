"""FFT local transformer.

Author: Sergey Ivanychev
"""
import numpy as np
import scipy.fftpack as fftpack

from . import base

class Fft(base.BaseLocalModel):
    def __init__(self) -> None:
        pass
    
    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        params = fftpack.fft(row)
        predicted = fftpack.ifft(params)
        return base.ApproxAndParams(predicted, params)


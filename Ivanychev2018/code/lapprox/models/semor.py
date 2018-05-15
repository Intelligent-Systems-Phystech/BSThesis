import numpy as np

from . import base

class Semor(base.BaseLocalModel):
    def __init__(self, profile: np.ndarray, time_array: np.ndarray):
        self._profile = profile
        self._profile_time = time_array

    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        pass
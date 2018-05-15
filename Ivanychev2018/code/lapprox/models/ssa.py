import numpy as np


from . import base
from ..misc import ssa_wrapper


class Ssa(base.BaseLocalModel):
    def __init__(self, lag: int):
        self._lag = lag

    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        ssa = ssa_wrapper.SSA(row, self._lag)
        return base.ApproxAndParams(ssa.get_predicted(),
                                    ssa.get_singular_values())
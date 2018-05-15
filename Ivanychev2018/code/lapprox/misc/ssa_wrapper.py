import numpy as np

from ..misc import utils
from ..thirdparty import ssa as third_ssa

class SSA(object):
    def __init__(self, X: np.ndarray, lag: int):
        self._X = X
        self._df_X = utils.array_to_date_indexed_df(X)
        self._lag = lag
        self.ssa = third_ssa.SSA(self._df_X)
        self.ssa.embed(embedding_dimension=lag)
        self.ssa.decompose()

    def get_singular_values(self) -> np.ndarray:
        return self.ssa.s

    def get_rank(self) -> int:
        return self.ssa.d

    def get_predicted(self) -> np.ndarray:
        rank = self.get_rank()
        res = self.ssa.view_reconstruction(*[self.ssa.Xs[i] for i in range(rank)],
                                             names=range(rank),
                                             return_df=True, plot=False)
        return res.as_matrix().flatten()

import datetime
import itertools

from typing import List


import numpy as np
import pandas as pd


def to_matrix(X: np.ndarray) -> np.ndarray:
    if len(X.shape) == 1:
        return X.reshape((1, -1))
    return X


def rolling(X: np.ndarray, window: int) -> np.ndarray:
    """ [1, 2, 3, 4, 5], 2 -> [[1, 2], [2, 3], [3, 4], [4, 5]]"""
    shape = (X.size - window + 1, window)
    strides = (X.itemsize, X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def array_to_date_indexed_df(x: np.ndarray) -> pd.DataFrame:
    start = datetime.datetime.fromtimestamp(0)
    index = pd.date_range(start=start, freq='H', periods=len(x), name="time")
    df = pd.DataFrame(x, index=index, columns=["values"])
    return df


def all_combinations(lst: List):
    comb_generator = itertools.chain.from_iterable(
        itertools.combinations(lst, size) for size in range(1, len(lst)))
    return list(comb_generator)
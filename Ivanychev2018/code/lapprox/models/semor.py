"""Self-Modeling Regression local transformer.

Author: Sergey Ivanychev
"""

import itertools

from typing import Callable, Optional, Tuple

import dtw
import numpy as np
import scipy
import sklearn.linear_model

from . import base


class Semor(base.BaseLocalModel):
    def __init__(self, shape: np.ndarray,
                 dist: Optional[Callable[[float, float], float]]) -> None:
        """Self-Modeling Regression.

        :param shape: np.ndarray. The shape (time series profile) array which is
            being compared against.
        :param dist: The distance function in the DTW algorithm.
        """
        self._dist = dist if dist else lambda x, y: np.linalg.norm(x - y, ord=1)
        self._shape = shape.reshape((-1, 1))

    def _dtw(self, row: np.ndarray) -> Tuple[float, float, float]:
        dist, cost, acc, path = dtw.dtw(row, self._shape, dist=self._dist)

        # Counting how many times we don't start going up.
        zeros = itertools.takewhile(lambda x: not x, path[1])
        print(f"Path[1]: {path[1]}")
        zeros_count = len(list(zeros))
        bottom_idx = zeros_count - 1

        # Finding out when we reached the top.
        top_idx = np.where(path[1] == len(self._shape) - 1)[0][0]
        decline = (top_idx - bottom_idx + 1) / float(len(self._shape))

        return float(bottom_idx), decline, dist

    def fit_row(self, row: np.ndarray) -> base.ApproxAndParams:
        row = row.reshape((-1, 1))

        # w_3, w_4
        shift, stretch, dist = self._dtw(row)
        print(f"Shift: {shift}, stretch: {stretch}")

        # We now find out which row indices within row will be used for
        # calculating norm of difference between the row and the stretched
        # profile.
        if stretch > 1:
            # In this case the profile is being stretched and thus we need to
            # interpolate it within itself.
            stretched_profile_len = np.floor(len(self._shape) * stretch)
            times = np.arange(shift,
                              min(shift + stretched_profile_len, len(row)),
                              dtype=np.int)
            row_part = row[times]
            x = np.linspace(shift,
                            shift + stretched_profile_len,
                            num=self._shape.size)
            y = self._shape.flatten()
            interpolated_shape = scipy.interpolate.interp1d(x, y, kind='cubic')
            shape_part = interpolated_shape(times)
        else:
            # In this case the profile is being squeezed and thus we need to
            # interpolate the row within the time boundaries of the squeezed
            # profile.
            shape_part = self._shape.reshape((-1,))
            times = shift + np.arange(len(self._shape)) * stretch

            x = np.arange(len(row))
            y = row.reshape((-1,))
            interpolated_row = scipy.interpolate.interp1d(x, y, kind='cubic')

            row_part = interpolated_row(times)
        row_part = row_part.flatten()
        shape_part = shape_part.flatten()

        # OK, we  have both row values and shape values. Now we need to find out
        # how we lift and scale the profile.

        rgr = sklearn.linear_model.LinearRegression()
        X = np.hstack((np.ones((shape_part.size, 1)),
                       shape_part.reshape((-1, 1))))
        rgr.fit(X, row_part)

        # w_1, w_2
        lift, scale = rgr.coef_
        return base.ApproxAndParams(
            row.flatten(), np.array([lift, scale, shift, stretch, dist])
        )

#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" Time-Series package """


import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


class TSDataset(object):
    """Dataset class

    Attributes:
        ts ... (m,) numpy.array, time-series
        label ... (m,) numpy.array, class labels
    """

    def __init__(self):
        self.__dataset = np.array([[]], dtype=[('ts', 'O'), ('label', 'O')])

    def __getattr__(self, attr):
        if attr == 'ts':
            return self.__dataset['ts'].ravel()
        elif attr == 'label':
            return np.hstack(self.__dataset['label'].ravel()).ravel()

        raise AttributeError("'TSDataset' object has no attribute '%s'" % attr)

    def __getitem__(self, i):
        return (self.ts[i], self.label[i])

    def __len__(self):
        return self.ts.size

    @classmethod
    def load_from_mat(cls, mat_file):
        dataset = cls()
        dataset.__dataset = sio.loadmat(mat_file)['dataset']
        return dataset

    def save_to_mat(self, mat_file, do_compression=True):
        if not os.path.exists(os.path.dirname(mat_file)):
            os.makedirs(os.path.dirname(mat_file))

        sio.savemat(mat_file, {'dataset': self.__dataset},
                    do_compression=do_compression)

    def add_observation(self, ts, label):
        self.__dataset = np.hstack((
            self.__dataset,
            np.array((ts, label), dtype=[('ts', 'O'), ('label', 'O')], ndmin=2)
        ))

    def extend(self, ts_list, label_list):
        for ts, label in zip(ts_list, label_list):
            self.add_observation(ts, label)


def ExtractFeatures(ts, *extractors):
    """Extract features from a given set of time-series

    Keyword arguments:
        ts ... (m,) numpy.array, set of time-series
        extractors ... callable, extract features from a single time-seres
    """
    if len(extractors) is 0:
        raise ValueError("Feature extractor was not specified")

    def get_features(ts):
        return np.hstack([extractor(ts) for extractor in extractors])
    
    def my_map(func, data):
        total = len(data)
        from IPython.display import clear_output
        import sys
        for i, item in enumerate(data):
            yield func(item)
            clear_output(wait=True)
            sys.stdout.write(str(((i + 1) * 100) // total) + "%")
            sys.stdout.flush()

    return np.vstack(my_map(get_features, ts))


def transform_frequency(time, ts, freq, kind='linear'):
    """1-D cubic spline interpolation

    Keyword arguments:
        time ... (t,) numpy.array, time points
        ts   ... (n, t) numpy.array, time-series
        freq ... frequency
        kind ... interpolation method: 'linear', 'quadratic', 'cubic'
    """
    t = np.linspace(time[0], time[-1], (time[-1] - time[0]) * freq + 1,
                    endpoint=True)
    ts_transformed = []
    for ts_axis in np.atleast_2d(ts):
        interpolation = interp1d(time, ts_axis.ravel(), kind)
        ts_transformed.append(interpolation(t))
    return t, np.vstack(ts_transformed)


def smooth(ts, window_size):
    ts_2d = np.atleast_2d(ts).copy()
    ts_2d = ts_2d[:, :ts_2d.shape[1] - (ts_2d.shape[1] % window_size)]
    return ts_2d.reshape(ts_2d.shape[0], -1, window_size).mean(2)

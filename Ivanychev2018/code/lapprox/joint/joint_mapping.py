from typing import Dict

import numpy as np
import sklearn.base

from ..models import base

class JointMapping(sklearn.base.TransformerMixin,
                   sklearn.base.BaseEstimator):
    def __init__(self, models: Dict[str, base.BaseLocalModel]):
        self._models = models

    def fit(self, X, y=None) -> 'JointMapping':
        for name, model in self._models.items():
            model.fit(X)
        return self

    def transform(self, X=None) -> np.ndarray:
        self._transformed = {name: model.transform()
                             for name, model in self._models.items()}
        last_index = 0
        for name, transformed in self._transformed.items():
            next_index = last_index + transformed.shape[1]
            print(f"Model {name}: indices {last_index}–{next_index}")
            last_index = next_index
        return np.concatenate(list(self._transformed.values()), axis=1)

    @property
    def models(self):
        return self._models
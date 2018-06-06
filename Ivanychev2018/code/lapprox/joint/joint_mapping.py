import sys
import traceback
from typing import Dict

import numpy as np
import sklearn.base

from ..models import base

class JointMapping(sklearn.base.TransformerMixin,
                   sklearn.base.BaseEstimator):
    def __init__(self,
                 models: Dict[str, base.BaseLocalModel],
                 fitted=None,
                 namespace_indices=None,
                 transformed=None):
        self.models = models
        self.fitted = fitted or {key: False for key in self.models}
        self.namespace_indices = namespace_indices
        self.transformed = transformed

    def fit(self, X, y=None) -> 'JointMapping':
        for name, model in self.models.items():
            try:
                model.fit(X)
            except:
                print("Warning: {} is not fitted due to error".format(name))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=10, file=sys.stdout)

            else:
                self.fitted[name] = True
        return self

    def transform(self, X=None, y=None) -> np.ndarray:
        if self.transformed is None:
            self.transformed = {}
            for name, model in self.models.items():
                if self.fitted[name]:
                    self.transformed[name] = model.transform()

        if self.namespace_indices is None:
            self.namespace_indices = {}
            last_index = 0
            for name, transformed in self.transformed.items():
                next_index = last_index + transformed.shape[1]
                self.namespace_indices[name] = np.arange(last_index,
                                                         next_index)
                last_index = next_index
        return np.concatenate(list(self.transformed.values()), axis=1)

    @property
    def namespaces(self):
        return list(self.transformed)


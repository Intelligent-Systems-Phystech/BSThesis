from typing import List

import numpy as np
import sklearn.base

from ..joint import joint_mapping


class CutNamespacesTransformer(sklearn.base.TransformerMixin,
                               sklearn.base.BaseEstimator):
    def __init__(self,
                 namespaces: List[str],
                 mapping: joint_mapping.JointMapping):
        self.mapping = mapping
        self.namespaces = namespaces

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        per_namespace_indices = tuple(self.mapping.namespace_indices[name]
                                      for name in self.namespaces)
        indices = np.hstack(per_namespace_indices)
        return X[:, indices]
from .Model import Model
import numpy as np
from econml.ortho_forest import ContinuousTreatmentOrthoForest

class OrthoForest(Model):
    def __init__(self):
        self.reg = None

    def fit(self, x, t, y, nfolds=5, seed=282):
        # splits = super().get_splits(x, nfolds, seed)
        self.reg = ContinuousTreatmentOrthoForest(n_trees=1, subsample_ratio=1)
        self.reg.fit(y, t, x)

    def predict(self, x, t):
        if self.reg is None:
            raise Exception('OrthoForest not Initialized')

        print("x", x.shape, x)
        print("t", t.shape, t)
        effect = self.reg.const_marginal_effect(x)
        print("effect", effect.shape, effect)
        return self.reg.const_marginal_effect(x)

    def get_predictors(self, x, t):
        return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
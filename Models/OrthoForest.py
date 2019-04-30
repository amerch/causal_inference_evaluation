from .Model import Model
import numpy as np
from econml.ortho_forest import ContinuousTreatmentOrthoForest, DiscreteTreatmentOrthoForest

class OrthoForest(Model):
    def __init__(self, *args, **kwargs):
        self.reg = None
        super(OrthoForest, self).__init__(*args, **kwargs)

    def fit(self, x, t, y, nfolds=5, seed=282):
        # splits = super().get_splits(x, nfolds, seed)
        #### CLASSIFICATION ####
    	if self.binary:
            self.reg = DiscreteTreatmentOrthoForest(n_trees=1,
                                max_depth=2,
                                n_jobs=100,
                                subsample_ratio=0.25,
                                random_state=282)
            self.reg.fit(y, t, x)

        #### REGRESSION ####
    	else: 
            self.reg = ContinuousTreatmentOrthoForest(n_trees=1,
                                max_depth=2,
                                n_jobs=100,
                                subsample_ratio=0.25,
                                random_state=282)
            self.reg.fit(y, t, x)

    def predict(self, x, t):
        if self.reg is None:
            raise Exception('OrthoForest not Initialized')

        # print("x", x.shape, x)
        # print("t", t.shape, t)
        effect = self.reg.const_marginal_effect(x).reshape(-1)
        # print("effect", effect.shape, effect)
        return effect * t

    def get_predictors(self, x, t):
        return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
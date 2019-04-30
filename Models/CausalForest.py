from .Model import Model
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

class CausalForest(Model):
    def __init__(self):
        self.reg = None

    def fit(self, x, t, y, nfolds=5, seed=282):
        # adapted from http://aeturrell.com/2018/03/28/estimation-heterogeneous-treatment-random-forests/
        n = x.shape[0]
        s = int(n/2)

        subSampleMask = random.sample(range(0, n), s)
        setIMask = random.sample(subSampleMask, np.int(np.ceil(s/2.)))
        setI = [x[setIMask]]

        setJMask = [i for i in subSampleMask if i not in setIMask]
        setJ = [x[setJMask]]

        # splits = super().get_splits(x, nfolds, seed)
        self.reg = RandomForestRegressor(max_depth=2,
                        n_estimators=100,
                        random_state=282,)\
                    .fit(self.get_predictors(x, t), y)

    def predict(self, x, t):
        if self.reg is None:
            raise Exception('CausalForest not Initialized')

        return self.reg.predict(self.get_predictors(x, t))

    def get_predictors(self, x, t):
        return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
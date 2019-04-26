from .Model import Model
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForest(Model):
    def __init__(self):
        self.reg = None

    def fit(self, x, t, y, nfolds=5, seed=282):
        # splits = super().get_splits(x, nfolds, seed)
        self.reg = RandomForestRegressor(max_depth=2,
                        n_estimators=100,
                        random_state=282,)\
                    .fit(self.get_predictors(x, t), y)

    def predict(self, x, t):
        if self.reg is None:
            raise Exception('RandomForest not Initialized')

        return self.reg.predict(self.get_predictors(x, t))

    def get_predictors(self, x, t):
        return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
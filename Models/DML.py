from .Model import Model
from econml.dml import DMLCateEstimator
from sklearn.linear_model import LassoCV, LinearRegression

class DML(Model):
    def __init__(self):
        self.reg = None

    def fit(self, x, t, y, nfolds=5, seed=282):
        # splits = super().get_splits(x, nfolds, seed)
        self.reg = DMLCateEstimator(model_y=LassoCV(), model_t=LassoCV, random_state=282)
        self.reg.fit(y, t, x)

    def predict(self, x, t):
        if self.reg is None:
            raise Exception('DML not Initialized')

        print("x", x.shape, x)
        print("t", t.shape, t)
        effect = self.reg.const_marginal_effect(x)
        print("effect", effect.shape, effect)
        return self.reg.const_marginal_effect(x)

    def get_predictors(self, x, t):
        return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
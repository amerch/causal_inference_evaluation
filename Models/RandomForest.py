from .Model import Model
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class RandomForest(Model):
	def __init__(self, *args, **kwargs):
		self.reg = None
		super(RandomForest, self).__init__(*args, **kwargs)

	def fit(self, x, t, y, nfolds=5, seed=282):
		# splits = super().get_splits(x, nfolds, seed)

		#### CLASSIFICATION ####
		if self.binary:
			self.reg = RandomForestClassifier(max_depth=3,
							n_estimators=100,
							random_state=282,)\
						.fit(self.get_predictors(x, t), y)

		#### REGRESSION ####
		else: 
			self.reg = RandomForestRegressor(max_depth=3,
							n_estimators=100,
							random_state=282,)\
						.fit(self.get_predictors(x, t), y)

	def predict(self, x, t):
		if self.reg is None:
			raise Exception('RandomForest not Initialized')

		#### CLASSIFICATION ####
		if self.binary:
			return self.reg.predict_proba(self.get_predictors(x, t))[:, 1]

		#### REGRESSION ####
		else:    
			# print("x", x.shape, x)
			# print("t", t.shape, t)
			effect = self.reg.predict(self.get_predictors(x, t))
			# print("effect", effect.shape, effect)
			return effect

	def get_predictors(self, x, t):
		return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
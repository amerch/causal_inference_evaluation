from .Model import Model
import numpy as np
from sklearn.linear_model import LinearRegression

class S_learner(Model):
	def __init__(self):
		self.reg = None

	def fit(self, x, t, y):
		x_cat = np.concatenate([x, t], axis=1)
		reg = LinearRegression().fit(x_cat, y)
		self.reg = reg

	def predict(self, x, t):
		if self.reg is None:
			raise Exception('S_learner not Initialized')

		x_cat = np.concatenate([x, t], axis=1)
		return self.reg.predict(x_cat)
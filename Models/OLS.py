from .Model import Model
import numpy as np
from sklearn.linear_model import LinearRegression

class OLS(Model):
	def __init__(self, *args, **kwargs):
		self.reg = None
		super(OLS, self).__init__(*args, **kwargs)

	def fit(self, x, t, y):
		t = t.reshape(-1, 1)
		x_cat = np.concatenate([x, t], axis=1)
		reg = LinearRegression().fit(x_cat, y)
		self.reg = reg

	def predict(self, x, t):
		if self.reg is None:
			raise Exception('OLS not Initialized')
			
		t = t.reshape(-1, 1)

		x_cat = np.concatenate([x, t], axis=1)
		return self.reg.predict(x_cat)

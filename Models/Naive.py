from .Model import Model
import numpy as np

class Naive(Model):
	def __init__(self, *args, **kwargs):
		self.y1 = None
		self.y0 = None
		super(Naive, self).__init__(*args, **kwargs)

	def fit(self, x, t, y):
		self.y0 = y[t==0].mean()
		self.y1 = y[t==1].mean()

	def predict(self, x, t):
		if self.y1 is None or self.y0 is None:
			raise Exception('Naive not Initialized')
		
		pred = np.zeros(t.shape)
		if sum(t == 0) != 0:
			pred[t==0] = self.y0

		if sum(t == 0) != t.shape[0]:
			pred[t==1] = self.y1

		return pred
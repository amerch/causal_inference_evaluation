class Model(object):
	def __init__(self):
		pass

	def fit(self, x, t, y):
		raise NotImplementedError

	def predict(self, x, t):
		raise NotImplementedError
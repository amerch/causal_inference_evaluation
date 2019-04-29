from .Model import Model
import numpy as np
from sklearn.linear_model import Lasso

class TLearner(Model):
	def __init__(self, *args, **kwargs):
		self.reg_treated = None
		self.reg_untreated = None
		self.l1_penalty = 0
		super(TLearner, self).__init__(*args, **kwargs)

	def fit(self, x, t, y, nfolds=5, lambdas=[1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3], seed=1234):
		splits = super().get_splits(x, nfolds, seed)
		avg_rmses = []
		for l in lambdas:
			rmse_lst = []
			for i, (train_index, valid_index) in enumerate(splits):
				x_train, x_valid = x[train_index], x[valid_index]
				t_train, t_valid = t[train_index], t[valid_index]
				y_train, y_valid = y[train_index], y[valid_index]
				treated = (t_train == 1)
				reg_treated = Lasso(alpha=l).fit(x_train[treated], y_train[treated])
				reg_untreated = Lasso(alpha=l).fit(x_train[~treated], y_train[~treated])
				yhat_valid = np.where(t_valid, reg_treated.predict(x_valid), 
					reg_untreated.predict(x_valid))
				rmse = np.sqrt(np.mean((yhat_valid - y_valid) ** 2))
				rmse_lst.append(rmse)
			avg_rmses.append(np.mean(rmse_lst, axis=0))
		self.l1_penalty = lambdas[np.argmin(avg_rmses)]
		treated = (t == 1)
		self.reg_treated = Lasso(alpha=self.l1_penalty).fit(x[treated], y[treated])
		self.reg_untreated = Lasso(alpha=self.l1_penalty).fit(x[~treated], y[~treated])

	def predict(self, x, t):
		if self.reg_treated is None or self.reg_untreated is None:
			raise Exception('TLearner not Initialized')

		return np.where(t == 1, self.reg_treated.predict(x), self.reg_untreated.predict(x))
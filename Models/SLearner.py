from .Model import Model
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

class SLearner(Model):
	def __init__(self, *args, **kwargs):
		self.reg = None
		self.l1_penalty = 0
		super(SLearner, self).__init__(*args, **kwargs)

	def fit(self, x, t, y, nfolds=5, lambdas=[1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3], seed=282):

		splits = super().get_splits(x, nfolds, seed)
		
		#### CLASSIFICATION ####
		if self.binary:
			avg_bces = []
			for l in lambdas:
				bce_lst = []
				for train_index, valid_index in splits:
					x_train, x_valid = x[train_index], x[valid_index]
					t_train, t_valid = t[train_index], t[valid_index]
					y_train, y_valid = y[train_index], y[valid_index]
					reg = LogisticRegression(C=1/l).fit(self.get_predictors(x_train, t_train), y_train)
					log_phat_valid = reg.predict_log_proba(self.get_predictors(x_valid, t_valid))
					bce = np.mean(log_phat_valid[np.arange(len(y_valid)), y_valid.astype('int')])
					bce_lst.append(bce)
				avg_bces.append(np.mean(bce_lst))
			self.l1_penalty = lambdas[np.argmin(avg_bces)]
			self.reg = LogisticRegression(C=1/self.l1_penalty).fit(self.get_predictors(x, t), y)

		#### REGRESSION ####
		else:
			avg_rmses = []
			for l in lambdas:
				rmse_lst = []
				for train_index, valid_index in splits:
					x_train, x_valid = x[train_index], x[valid_index]
					t_train, t_valid = t[train_index], t[valid_index]
					y_train, y_valid = y[train_index], y[valid_index]
					reg = Lasso(alpha=l).fit(self.get_predictors(x_train, t_train), y_train)
					yhat_valid = reg.predict(self.get_predictors(x_valid, t_valid))
					rmse = np.sqrt(np.mean((yhat_valid - y_valid) ** 2))
					rmse_lst.append(rmse)
				avg_rmses.append(np.mean(rmse_lst))
			self.l1_penalty = lambdas[np.argmin(avg_rmses)]
			self.reg = Lasso(alpha=self.l1_penalty).fit(self.get_predictors(x, t), y)

	def predict(self, x, t):
		if self.reg is None:
			raise Exception('SLearner not Initialized')

		#### CLASSIFICATION ####
		if self.binary:
			return self.reg.predict_proba(self.get_predictors(x, t))[:, 1]

		#### REGRESSION ####
		else:
			return self.reg.predict(self.get_predictors(x, t))

	def get_predictors(self, x, t):
		return np.hstack([x, (t - 0.5).reshape(-1, 1) * x])
from .Model import Model
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

class KNN(Model):
	def __init__(self, *args, **kwargs):
		self.reg = None
		super(KNN, self).__init__(*args, **kwargs)

	def fit(self, x, t, y):
		x_control, y_control = x[t == 0], y[t == 0]
		x_treat, y_treat = x[t == 1], y[t == 1]

		#### CLASSIFICATION #### 
		if self.binary:
			reg_control = KNeighborsClassifier().fit(x_control, y_control)
			reg_treat = KNeighborsClassifier().fit(x_treat, y_treat)
		else:	
			reg_control = KNeighborsRegressor().fit(x_control, y_control)
			reg_treat = KNeighborsRegressor().fit(x_treat, y_treat)
		
		self.reg_control = reg_control
		self.reg_treat = reg_treat

	def predict(self, x, t):
		if self.reg_control is None or self.reg_treat is None:
			raise Exception('OLS Separate not Initialized')
		

		pred = np.zeros(t.shape)

		if self.binary:
			if sum(t == 0) != 0:
				pred[t==0] = self.reg_control.predict_proba(x[t==0])[:,1]

			if sum(t == 0) != t.shape[0]:
				pred[t==1] = self.reg_treat.predict_proba(x[t==1])[:, 1]
			return pred

		else:
			if sum(t == 0) != 0:
				pred[t==0] = self.reg_control.predict(x[t==0])

			if sum(t == 0) != t.shape[0]:
				pred[t==1] = self.reg_treat.predict(x[t==1])
	
		return pred

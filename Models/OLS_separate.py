from .Model import Model
import numpy as np
from sklearn.linear_model import LinearRegression

class OLS_separate(Model):
	def __init__(self):
		self.reg = None

	def fit(self, x, t, y):
                x_control, y_control = x[t == 0], y[t == 0]
                x_treat, y_treat = x[t == 1], y[t == 1]

                reg_control = LinearRegression().fit(x_control, y_control)
                reg_treat = LinearRegression().fit(x_treat, y_treat)
		
                self.reg_conrol = reg_control
                self.reg_treat = reg_treat

	def predict(self, x, t):
                if self.reg_control is None or self.reg_treat is None:
			raise Exception('S_learner not Initialized')
                
                pred = np.zeros(t.shape)
                pred[t==0] = self.reg_control.predict(x[t==0])
                pred[t==1] = self.reg_treat.predict(x[t==1])
	
                return pred

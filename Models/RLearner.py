from .Model import Model
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

class RLearner(Model):
  def __init__(self):
    self.marg_resp = None
    self.marg_resp_l1_penalty = None
    self.prop = None
    self.prop_l1_penalty = None
    self.rlearner = None
    self.rlearner_penalty = None

  def fit(self, x, t, y, nfolds=5, lambdas=[1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3], seed=1234):
    splits = super().get_splits(x, nfolds, seed)

    # fit marginal response model
    avg_rmses = []
    for l in lambdas:
      rmse_lst = []
      for train_index, valid_index in splits:
        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        marg_resp = Lasso(alpha=l).fit(x_train, y_train)
        yhat_valid = marg_resp.predict(x_valid)
        rmse = np.sqrt(np.mean((yhat_valid - y_valid) ** 2))
        rmse_lst.append(rmse)
      avg_rmses.append(np.mean(rmse_lst))
    self.marg_resp_l1_penalty = lambdas[np.argmin(avg_rmses)]

    # fit propensity model
    avg_bin_cross_entropys = []
    for l in lambdas:
      bin_cross_entropy_lst = []
      for train_index, valid_index in splits:
        x_train, x_valid = x[train_index], x[valid_index]
        t_train, t_valid = t[train_index], t[valid_index]
        prop = LogisticRegression(penalty='l1', C=1/l).fit(x_train, t_train)
        that_means = prop.predict_log_proba(x_valid)
        n_valid = that_means.shape[0]
        bin_cross_entropy = np.sum(-that_means[np.arange(n_valid), t_valid.astype('int')])
        bin_cross_entropy_lst.append(bin_cross_entropy)
      avg_bin_cross_entropys.append(np.mean(bin_cross_entropy_lst))
    self.prop_l1_penalty = lambdas[np.argmin(avg_bin_cross_entropys)]

    # collect explanatory and response for RLearner
    N = x.shape[0]
    explanatory = np.empty(N)
    response = np.empty(N)
    for i in range(N):
      x_rest = np.vstack((x[:i], x[(i+1):]))
      y_rest = np.concatenate((y[:i], y[(i+1):]))
      t_rest = np.concatenate((t[:i], t[(i+1):]))
      marg_resp = Lasso(alpha=self.marg_resp_l1_penalty).fit(x_rest, y_rest)
      prop = LogisticRegression(penalty='l1', C=1/self.prop_l1_penalty).fit(x_rest, t_rest)
      explanatory[i] = t[i] - prop.predict_proba(x[i].reshape(1, -1))[0, 1]
      response[i] = y[i] - marg_resp.predict(x[i].reshape(1, -1))

    # fit RLearner
    avg_rmses = []
    for l in lambdas:
      rmse_lst = []
      for train_index, valid_index in splits:
        expl_train, expl_valid = explanatory[train_index], explanatory[valid_index]
        resp_train, resp_valid = response[train_index], response[valid_index]
        rlearner = Lasso(alpha=l).fit(expl_train.reshape(-1, 1), resp_train)
        resphat_valid = rlearner.predict(expl_valid.reshape(-1, 1))
        rmse = np.sqrt(np.mean((resphat_valid - resp_valid) ** 2))
        rmse_lst.append(rmse)
      avg_rmses.append(np.mean(rmse_lst))
    self.rlearner_penalty = lambdas[np.argmin(avg_rmses)]

    # fit all models
    self.marg_resp = Lasso(alpha=self.marg_resp_l1_penalty).fit(x, y)
    self.prop = LogisticRegression(penalty='l1', C=1/self.prop_l1_penalty).fit(x, t)
    x_const = np.hstack([np.ones(N).reshape(-1, 1), x])
    explanatory = ((t - self.prop.predict_proba(x)[:, 1]) * x_const.T).T
    response = y - self.marg_resp.predict(x)
    self.rlearner = Lasso(alpha=self.rlearner_penalty, fit_intercept=False).fit(explanatory, response)

  def predict(self, x, t):
    if self.rlearner is None:
      raise Exception('RLearner not Initialized')

    N = x.shape[0]
    marg_resp = self.marg_resp.predict(x)
    prop = self.prop.predict_proba(x)[:, 1]
    x_const = np.hstack([np.ones(N).reshape(-1, 1), x])
    cate = x_const @ self.rlearner.coef_ 
    
    return (t - prop) * cate + marg_resp
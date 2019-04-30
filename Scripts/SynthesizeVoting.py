import numpy as np
import pandas as pd

# Synthesizes Voting Data
# Adapted from https://github.com/xnie/rlearner/blob/master/experiments_for_paper/section2_example/run_analysis.R

np.random.seed(282)
df = pd.read_csv('../Data/Voting/data_clean.csv')
n_total = df.shape[0]
n_train = 100000
n_test = 25000
n_holdout = df.shape[0] - n_train - n_test
indices = np.arange(n_total)

train = {
  'x': [],
  'yf': [],
  'ycf': [],
  't': [],
  'ite': []
}

test = {
  'x': [],
  'yf': [],
  'ycf': [],
  't': [],
  'ite': []
}

holdout = {
  'x': [],
  'yf': [],
  'ycf': [],
  't': [],
  'ite': []
}

for i in range(8):

  print(i)

  # Make Synthetic Treatment
  df = df.iloc[indices]
  X = df[df.columns[:11]] # Covariates
  y_original = df['Y']
  t = df['W'] # Treatment
  tau = -X['vote00'] * 0.5 / (1 + 50 / X['age']) # CATE
  flip = np.random.binomial(n=1, p=np.abs(tau))

  y_outcomes = np.empty((n_total, 2))
  flip_mask = (flip == 1)
  tau_mask = (tau > 0)
  y_outcomes[~flip_mask, 0] = y_original[~flip_mask]
  y_outcomes[~flip_mask, 1] = y_original[~flip_mask]
  y_outcomes[flip_mask & tau_mask, 0] = 0
  y_outcomes[flip_mask & tau_mask, 1] = 1
  y_outcomes[flip_mask & ~tau_mask, 0] = 1
  y_outcomes[flip_mask & ~tau_mask, 1] = 0
  y = y_outcomes[np.arange(n_total), t]
  ycf = y_outcomes[np.arange(n_total), 1-t]

  X = X.values
  t = t.values
  tau = tau.values

  # Train data
  X_train = X[:n_train]
  t_train = t[:n_train]
  y_train = y[:n_train]
  ycf_train = ycf[:n_train]
  tau_train = tau[:n_train]

  # Test data
  X_test = X[n_train:n_train+n_test]
  t_test = t[n_train:n_train+n_test]
  y_test = y[n_train:n_train+n_test]
  ycf_test = ycf[n_train:n_train+n_test]
  tau_test = tau[n_train:n_train+n_test]

  # Holdout data
  X_holdout = X[n_train+n_test:n_train+n_test+n_holdout]
  t_holdout = t[n_train+n_test:n_train+n_test+n_holdout]
  y_holdout = y[n_train+n_test:n_train+n_test+n_holdout]
  ycf_holdout = ycf[n_train+n_test:n_train+n_test+n_holdout]
  tau_holdout = tau[n_train+n_test:n_train+n_test+n_holdout]

  # Save data
  train['x'].append(X_train.copy())
  train['t'].append(t_train.copy())
  train['yf'].append(y_train.copy())
  train['ycf'].append(ycf_train.copy())
  train['ite'].append(tau_train.copy())

  test['x'].append(X_test.copy())
  test['t'].append(t_test.copy())
  test['yf'].append(y_test.copy())
  test['ycf'].append(ycf_test.copy())
  test['ite'].append(tau_test.copy())

  holdout['x'].append(X_holdout.copy())
  holdout['t'].append(t_holdout.copy())
  holdout['yf'].append(y_holdout.copy())
  holdout['ycf'].append(ycf_holdout.copy())
  holdout['ite'].append(tau_holdout.copy())

  # Shuffle dataset for next train-test-holdout split; first replication is always original dataset
  np.random.shuffle(indices)

for k in train:
  train[k] = np.stack(train[k], axis=-1)
for k in test:
  test[k] = np.stack(test[k], axis=-1)
for k in holdout:
  holdout[k] = np.stack(holdout[k], axis=-1)

np.savez('../Data/Voting/train.npz', **train)
np.savez('../Data/Voting/test.npz', **test)
np.savez('../Data/Voting/holdout.npz', **holdout)
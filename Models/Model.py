from sklearn.model_selection import KFold

class Model(object):
  def __init__(self):
    pass

  def fit(self, x, t, y):
    raise NotImplementedError

  def predict(self, x, t):
    raise NotImplementedError

  def get_splits(self, x, nfolds, seed):
    self.nfolds = nfolds
    self.seed = seed
    kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=self.seed)
    splits = kf.split(x)
    return list(splits)
 ## FROM CEVAE code with minor changes

import numpy as np

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None, true_ite=None, true_ate=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        self.true_ite = true_ite
        self.true_ate = true_ate

        if self.true_ite is None and self.mu0 is not None and self.mu1 is not None:
            self.true_ite = self.mu1 - self.mu0

        if self.true_ate is None and self.true_ite is not None:
            self.true_ate = np.mean(self.true_ite)

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1)[0], np.where(self.t == 0)[0]
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        pred_ate = np.mean(ypred1 - ypred0)
        return np.abs(pred_ate - self.true_ate)

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square(self.true_ite - (ypred1 - ypred0))))
            
    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):

        stats_names = []
        stats_vals = []

        if self.true_ite is not None:

            ite = self.rmse_ite(ypred1, ypred0)
            stats_names.append('RMSE_ITE')
            stats_vals.append(ite)

        if self.true_ate is not None:
            ate = self.abs_ate(ypred1, ypred0)
            stats_names.append('ABS_ATE')
            stats_vals.append(ate)

        if self.true_ite is not None:
            pehe = self.pehe(ypred1, ypred0)
            stats_names.append('PEHE')
            stats_vals.append(pehe)

        return stats_names, stats_vals
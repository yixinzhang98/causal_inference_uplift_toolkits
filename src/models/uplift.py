from __future__ import annotations
import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

class TLearnerUplift(BaseEstimator):
    """
    Two separate regressors: one for treated outcomes mu1(x) and one for control mu0(x).
    uplift(x) = mu1(x) - mu0(x)
    """
    def __init__(self, base_estimator=None, random_state=42):
        self.random_state = random_state
        if base_estimator is not None:
            self.model_t = base_estimator
            self.model_c = base_estimator
        else:
            if _HAS_LGBM:
                self.model_t = lgb.LGBMRegressor(random_state=random_state, n_estimators=300)
                self.model_c = lgb.LGBMRegressor(random_state=random_state, n_estimators=300)
            else:
                self.model_t = RandomForestRegressor(n_estimators=300, random_state=random_state)
                self.model_c = RandomForestRegressor(n_estimators=300, random_state=random_state)

    def fit(self, X, t, y):
        X = np.asarray(X)
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()
        self.model_t.fit(X[t == 1], y[t == 1])
        self.model_c.fit(X[t == 0], y[t == 0])
        return self

    def predict_mu1(self, X):
        return self.model_t.predict(X)

    def predict_mu0(self, X):
        return self.model_c.predict(X)

    def predict_uplift(self, X):
        return self.predict_mu1(X) - self.predict_mu0(X)


class CausalForestUplift(BaseEstimator):
    """
    Wrapper for econml's CausalForestDML if installed. Falls back to TLearner otherwise.
    """
    def __init__(self, random_state=42):
        try:
            from econml.dml import CausalForestDML  # type: ignore
            from sklearn.ensemble import RandomForestRegressor
            self._is_cf = True
            self.model = CausalForestDML(
                model_t=RandomForestRegressor(n_estimators=200, random_state=random_state),
                model_y=RandomForestRegressor(n_estimators=200, random_state=random_state),
                discrete_treatment=True,
                random_state=random_state
            )
        except Exception:
            self._is_cf = False
            self.model = TLearnerUplift(random_state=random_state)

    def fit(self, X, t, y):
        X = np.asarray(X)
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()
        if self._is_cf:
            self.model.fit(y, t, X=X)
        else:
            self.model.fit(X, t, y)
        return self

    def predict_uplift(self, X):
        if self._is_cf:
            te = self.model.effect(X)
            return te
        else:
            return self.model.predict_uplift(X)

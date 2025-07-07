import pandas as pd
from causalml.dataset import simulate_nuisance_and_easy_treatment
from causalml.inference.meta import LRSRegressor

def estimate_treatment_effect(X, treatment, y):
    lr = LRSRegressor()
    return lr.estimate_ate(X=X, treatment=treatment, y=y)

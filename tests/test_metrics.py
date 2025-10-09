import numpy as np
from pc_uplift.metrics.policy import qini_auc, auuc, policy_value

def test_basic_metrics_shapes():
    y = np.array([0,1,0,1,1,0,0,1])
    t = np.array([1,0,1,0,1,0,1,0])
    u = np.linspace(0,1,8)
    assert np.isfinite(qini_auc(y,t,u))
    assert np.isfinite(auuc(y,t,u))
    assert np.isfinite(policy_value(y,t,u,500,15,0.25))

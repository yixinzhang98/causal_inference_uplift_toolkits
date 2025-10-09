from __future__ import annotations
import numpy as np

def _order_by_uplift(uplift):
    return np.argsort(-np.asarray(uplift))

def qini_curve(y, t, uplift):
    y = np.asarray(y).ravel()
    t = np.asarray(t).ravel()
    uplift = np.asarray(uplift).ravel()
    order = _order_by_uplift(uplift)
    y_ord, t_ord = y[order], t[order]
    n = len(y)
    treated_rate = t.mean()
    gains = []
    treated_cum = 0
    control_cum = 0
    for k in range(1, n+1):
        if t_ord[k-1] == 1:
            treated_cum += y_ord[k-1]
        else:
            control_cum += y_ord[k-1]
        exp_control = (treated_cum + control_cum) * treated_rate
        gains.append(treated_cum - exp_control)
    return np.array(gains)

def qini_auc(y, t, uplift):
    q = qini_curve(y, t, uplift)
    return np.trapz(q / (len(y)), dx=1.0/len(y))

def auuc(y, t, uplift):
    q = qini_curve(y, t, uplift)
    return q.sum() / (len(y) ** 2)

def uplift_at_k(y, t, uplift, k=0.2):
    n = len(y)
    m = int(np.ceil(n * k))
    order = _order_by_uplift(uplift)[:m]
    return (y[order][t[order]==1].mean() - y[order][t[order]==0].mean())

def policy_value(y, t, uplift, benefit=500.0, cost=15.0, budget=0.2):
    n = len(y)
    m = int(np.ceil(n * budget))
    idx = _order_by_uplift(uplift)[:m]
    avg_upl = np.maximum(np.mean(np.asarray(uplift)[idx]), 0.0)
    ev = avg_upl * benefit * m - cost * m
    return float(ev)

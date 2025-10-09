import numpy as np
import pandas as pd

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def simulate_cohort(n=5000, seed=42):
    """
    Simulate a patient cohort with heterogeneous treatment effects for a
    reminder outreach on medication adherence.
    Returns a DataFrame with features X, treatment t, and outcome y (1=adherent).
    """
    rng = np.random.default_rng(seed)
    # Demographics / clinical features
    age = rng.normal(55, 12, n).clip(18, 90)
    female = rng.integers(0, 2, n)
    comorb = rng.poisson(1.5, n)  # count of chronic conditions
    prior_adherence = rng.beta(2, 2, n)  # historic proportion adherent
    risk_score = _sigmoid(-2 + 0.02*(age-50) + 0.4*female + 0.3*comorb - 2*(prior_adherence-0.5))

    # Subgroup (for fairness checks)
    region = rng.choice(["NE", "MW", "S", "W"], size=n, p=[0.28, 0.22, 0.30, 0.20])

    # True baseline and uplift functions
    mu0 = _sigmoid(-1.2 + 1.5*prior_adherence - 0.6*risk_score + 0.15*comorb)
    # Heterogeneous treatment effect: larger for lower prior adherence & higher risk
    tau = 0.10 + 0.25*(1 - prior_adherence) + 0.10*risk_score + 0.05*(region == "S")

    # Propensity with selection bias (more outreach for high-risk, low adherence)
    logit_p = -0.2 + 0.8*risk_score - 1.0*prior_adherence + 0.1*comorb + 0.1*(region=="NE")
    p_treat = _sigmoid(logit_p)
    t = rng.binomial(1, p_treat)

    # Realized outcome
    p_y = np.clip(mu0 + t * tau, 0, 1)
    y = rng.binomial(1, p_y)

    df = pd.DataFrame({
        "age": age,
        "female": female,
        "comorb": comorb,
        "prior_adherence": prior_adherence,
        "risk_score": risk_score,
        "region": region,
        "t": t,
        "y": y
    })
    return df

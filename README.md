# Predictive‑Causal Uplift Model for Patient Medication Reminders

> Production‑ready starter repo for modeling **incremental impact** (uplift) of reminder interventions on medication adherence.
> Includes synthetic healthcare data generator, uplift learners (T‑Learner, Causal Forest—optional), policy targeting,
> evaluation via Qini/AUUC, basic fairness checks, and a reproducible quickstart script.

## Why uplift (predictive‑causal) modeling?
Traditional predictive models estimate **risk** (e.g., “non‑adherence probability”). Uplift models estimate **change caused by an intervention**:
\[ U(x) = E[Y | T=1, X=x] - E[Y | T=0, X=x] \]
So you target **patients who will likely adhere because of outreach**, not those who would adhere anyway.

## Features
- **Synthetic cohort simulator** (no PHI): configurable treatment assignment bias, heterogeneous treatment effects.
- **Uplift learners**: T‑Learner (any sklearn regressor), optional *CausalForestDML* (econml, if installed).
- **Policy targeting**: top‑K or budget‑constrained selection; expected value with benefit/cost.
- **Evaluation**: Qini curve, AUUC, uplift@K, policy value.
- **Fairness & equity**: simple disparity checks across subgroups.
- **Reproducible**: one‑command `python run_experiment.py`.

## Quickstart
```bash
# 1) Create environment (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run an end‑to‑end example (train, evaluate, plot, policy simulation)
python run_experiment.py --n 15000 --budget 0.25 --benefit 500 --cost 15
```

This will print metrics (Qini, AUUC, uplift@K, policy value) and save artifacts to `artifacts/`.

## Repo layout
```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── run_experiment.py
├── src/pc_uplift/...
├── tests/...
└── docs/MODEL_CARD.md
```

## Minimal example (API)
```python
from pc_uplift.data.simulate import simulate_cohort
from pc_uplift.models.uplift import TLearnerUplift
from pc_uplift.metrics.policy import qini_auc, policy_value

df = simulate_cohort(n=5000, seed=42)
X = df.drop(columns=["y", "t"])
t = df["t"]
y = df["y"]

model = TLearnerUplift()  # defaults to RandomForest if LightGBM not installed
model.fit(X, t, y)
upl = model.predict_uplift(X)

print("Qini AUC:", qini_auc(y, t, upl))
print("Policy value (20% budget):", policy_value(y, t, upl, benefit=500, cost=15, budget=0.2))
```


# Causal Meta-Learner for Personalized Messaging

This repository implements a meta-learner architecture that integrates causal inference techniques to determine the most effective message for individual users.

## 🔍 Overview

- Estimate Conditional Average Treatment Effects (CATE)
- Evaluate uplift using meta-learner (e.g., S-learner, T-learner, X-learner)
- Apply to personalized messaging or marketing

## 🧠 Methods

- Meta-learners: S-Learner, T-Learner, X-Learner
- Causal estimators: Propensity Score Matching, Doubly Robust, Causal Forest
- Model options: XGBoost, Logistic Regression, MLP

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

## 📁 Structure

- `models/`: Base and meta learners
- `causal/`: Causal inference components
- `utils/`: Preprocessing utilities
- `experiments/`: Scripts for experiments
- `notebooks/`: Exploratory and usage examples

## 📊 Example Use Case

```bash
python experiments/run_experiment.py --learner x_learner --dataset data/sample_data.csv
```

## 📄 License

MIT License

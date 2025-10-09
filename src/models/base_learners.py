import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_base_learner(model_type='logistic'):
    if model_type == 'logistic':
        return LogisticRegression()
    elif model_type == 'random_forest':
        return RandomForestClassifier()
    else:
        raise ValueError("Unsupported model type")

# This function returns a base learner model based on the specified type.
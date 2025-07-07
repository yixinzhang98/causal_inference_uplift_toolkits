from causalml.inference.meta import XLearner, TLearner, SLearner
from sklearn.ensemble import GradientBoostingClassifier

def get_meta_learner(method='x_learner'):
    if method == 'x_learner':
        return XLearner(learner=GradientBoostingClassifier())
    elif method == 't_learner':
        return TLearner(learner=GradientBoostingClassifier())
    elif method == 's_learner':
        return SLearner(learner=GradientBoostingClassifier())
    else:
        raise ValueError("Unsupported learner type")

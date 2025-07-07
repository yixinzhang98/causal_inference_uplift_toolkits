from causalml.inference.tree import UpliftTreeClassifier

def get_uplift_model():
    return UpliftTreeClassifier(max_depth=5, min_samples_leaf=100)

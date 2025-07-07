import argparse
import pandas as pd
from utils.preprocess import preprocess_data
from models.meta_learner import get_meta_learner

def main(args):
    df = pd.read_csv(args.dataset)
    X, treatment, y = preprocess_data(df)

    learner = get_meta_learner(method=args.learner)
    learner.fit(X=X.values, treatment=treatment.values, y=y.values)

    te = learner.predict(X=X.values)
    print(f"Average Estimated Treatment Effect: {te.mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", type=str, default="x_learner", help="Meta learner type")
    parser.add_argument("--dataset", type=str, default="data/sample_data.csv", help="Path to dataset")
    args = parser.parse_args()
    main(args)

# This script runs a meta-learner on a specified dataset to estimate treatment effects.
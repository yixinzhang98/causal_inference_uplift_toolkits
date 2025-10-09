import argparse
import os
import numpy as np
import pandas as pd
from pc_uplift.data.simulate import simulate_cohort
from pc_uplift.models.uplift import TLearnerUplift, CausalForestUplift
from pc_uplift.metrics.policy import qini_auc, auuc, uplift_at_k, policy_value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="Cohort size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default="t", choices=["t", "cf"])
    ap.add_argument("--budget", type=float, default=0.2, help="Targeting fraction 0-1")
    ap.add_argument("--benefit", type=float, default=500.0, help="Benefit per adherence")
    ap.add_argument("--cost", type=float, default=15.0, help="Cost per outreach")
    args = ap.parse_args()

    df = simulate_cohort(n=args.n, seed=args.seed)
    X = pd.get_dummies(df.drop(columns=["y", "t"]), drop_first=True)
    t = df["t"].values
    y = df["y"].values

    if args.model == "cf":
        model = CausalForestUplift(random_state=args.seed)
    else:
        model = TLearnerUplift(random_state=args.seed)

    model.fit(X, t, y)
    upl = model.predict_uplift(X)

    qini = qini_auc(y, t, upl)
    a = auuc(y, t, upl)
    u20 = uplift_at_k(y, t, upl, k=0.2)
    pv = policy_value(y, t, upl, benefit=args.benefit, cost=args.cost, budget=args.budget)

    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/uplift.npy", upl)
    df_out = pd.DataFrame({"uplift": upl}).join(df)
    df_out.to_csv("artifacts/preds.csv", index=False)

    print(f"Qini AUC: {qini:.6f}")
    print(f"AUUC: {a:.6f}")
    print(f"Uplift@20%: {u20:.6f}")
    print(f"Policy value (budget={args.budget:.0%}): ${pv:,.2f}")
    print("Artifacts written to ./artifacts")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

df = pd.read_csv("stats.csv")
df["cv_rel"] = df["std_est"] / df["mean_exact"]


df["ok_cv_104"] = df["cv_rel"] <= df["theory_104"]
df["ok_cv_13"]  = df["cv_rel"] <= df["theory_13"]
df["ok_rmse_13"] = df["rmse_rel"] <= df["theory_13"]


summary = {
    "steps_count": len(df),

    "mean_abs_bias_rel": float(df["bias_rel"].abs().mean()),
    "max_abs_bias_rel": float(df["bias_rel"].abs().max()),

    "mean_rmse_rel": float(df["rmse_rel"].mean()),
    "max_rmse_rel": float(df["rmse_rel"].max()),

    "mean_cv_rel": float(df["cv_rel"].mean()),
    "max_cv_rel": float(df["cv_rel"].max()),

    "share_cv_below_104": float(df["ok_cv_104"].mean()),
    "share_cv_below_13":  float(df["ok_cv_13"].mean()),
    "share_rmse_below_13": float(df["ok_rmse_13"].mean()),
}

print("SUMMARY")
for k, v in summary.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

worst = df.sort_values("rmse_rel", ascending=False).head(10)[
    ["step_id", "processed", "mean_exact", "mean_est", "bias_rel", "rmse_rel", "cv_rel", "theory_104", "theory_13"]
]
worst.to_csv("analysis_worst_steps.csv", index=False)
print("\nSaved: analysis_worst_steps.csv")

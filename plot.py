import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("stats.csv")

# График 1
plt.figure(figsize=(9, 5))
plt.plot(df["processed"], df["mean_exact"], label="True F0_t")
plt.plot(df["processed"], df["mean_est"], label="HLL estimate N_t")
plt.xlabel("Processed elements (t)")
plt.ylabel("Unique elements")
plt.title("Graph #1: True F0_t vs HLL estimate N_t")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph1_compare.png", dpi=150)
plt.close()

# График 2
upper = df["mean_est"] + df["std_est"]
lower = df["mean_est"] - df["std_est"]

plt.figure(figsize=(9, 5))
plt.plot(df["processed"], df["mean_est"], label="E(N_t)")
plt.fill_between(df["processed"], lower, upper, alpha=0.25, label="E(N_t) ± σ_t")
plt.xlabel("Processed elements (t)")
plt.ylabel("Estimated unique elements")
plt.title("Graph #2: Mean estimate and uncertainty band")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph2_uncertainty.png", dpi=150)
plt.close()


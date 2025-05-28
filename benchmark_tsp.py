from tsp_problem import run_tsp_algorithms
import time
import pandas as pd

def run_multiple_tsp_trials(problem_sizes=[50,100,200,400], n_trials=200):
    records = []

    for size in problem_sizes:
        for trial in range(n_trials):
            seed = 42 + trial
            results = run_tsp_algorithms(seed, size)
            for algorithm, metrics in results.items():
                records.append({
                    "trial": trial,
                    "problem_size": size,
                    "algorithm": algorithm,
                    "cost_km": float(metrics["Distance (km)"]),
                    "runtime_s": float(metrics["Time (s)"])
                })

    df = pd.DataFrame(records)
    return df

# Run and collect data
df_results = run_multiple_tsp_trials()

# Save to CSV
df_results.to_csv("tsp_results.csv", index=False)

print("Results saved to tsp_results.csv")
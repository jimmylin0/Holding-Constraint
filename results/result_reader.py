import pandas as pd
import ast
import csv
import matplotlib.pyplot as plt

# Read existing file
with open(r"C:\Users\linj37\OneDrive - Lancaster University\Documents\Shell_EPO\Holding-Constraint\results\results.csv", "r") as f:
    reader = csv.reader(f)
    rows = list(reader)

results = [
    ast.literal_eval(cell)
    for row in rows[1:]
    for cell in row
    if cell.strip()
]

# Create a normal pandas DataFrame
df = pd.DataFrame(results)

df_filtered = df[
    (df["spread"] == 0) &
    (df["num_assets"] == 40) &
    (df["time_horizon"] == 200)
].copy()

#Plotting
# Columns defining each optimisation configuration
settings = ["bound_preprocess", "tighter_M", "variable_bound"]

# Calculate mean runtime for each holding time and configuration
plot_df = (
    df_filtered.groupby(["min_hold"] + settings, as_index=False)
      ["runtime"]
      .mean()
)

# Create plot
plt.figure(figsize=(11, 6))

# Plot each combination as a separate line
for config, group in plot_df.groupby(settings):

    label = (
        f"BP={config[0]}, "
        f"TM={config[1]}, "
        f"VB={config[2]}"
    )

    group = group.sort_values("min_hold")

    plt.plot(
        group["min_hold"],
        group["runtime"],
        label=label
    )

plt.xlabel("Minimum Holding Time")
plt.ylabel("Mean Runtime (seconds)")
plt.title("Runtime vs Minimum Holding Time")
plt.legend(title="Optimisation Settings", bbox_to_anchor=(1.05, 1),
           loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


######################
#relative plotting
# Filter the data

# Define optimisation settings
settings = ["bound_preprocess", "tighter_M", "variable_bound"]

# Average runtime across the 5 seeds
plot_df = (
    df_filtered.groupby(["min_hold"] + settings, as_index=False)
    ["runtime"].mean()
)

# Extract baseline (all settings False)
baseline = plot_df[
    (plot_df["bound_preprocess"] == False) &
    (plot_df["tighter_M"] == False) &
    (plot_df["variable_bound"] == False)
][["min_hold", "runtime"]]

baseline = baseline.rename(columns={"runtime": "baseline_runtime"})

# Match each configuration with its baseline at the same holding time
plot_df = plot_df.merge(
    baseline,
    on="min_hold",
    how="left",
    validate="many_to_one"
)

# Calculate runtime relative to baseline (%)
plot_df["relative_runtime"] = (
    plot_df["runtime"] / plot_df["baseline_runtime"]
) * 100

# Plot
plt.figure(figsize=(11, 6))

for config, group in plot_df.groupby(settings):

    group = group.sort_values("min_hold")

    label = (
        f"BP={config[0]}, "
        f"TM={config[1]}, "
        f"VB={config[2]}"
    )

    plt.plot(
        group["min_hold"],
        group["relative_runtime"],
        label=label
    )

# Baseline reference line
plt.axhline(100, color="black", linestyle="--", alpha=0.6)

plt.xlabel("Minimum Holding Time")
plt.ylabel("Relative Runtime (%)")
plt.ylim(0, 200)
plt.title("Relative Runtime vs Minimum Holding Time\n"
          "Spread = 0, Assets = 40, Horizon = 200")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
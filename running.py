import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from experimenter import experiment
from optimisation import det_ef_opt, generate_parameters, Parameters

#running the experiment file
time_horizon = 200
# h_list = [i for i in range(time_horizon)]
h_list = [1, 2, 4, 8, 16, 32, 64, 128]

#runtime storing
runtime = []
lp_gap = []

for h in h_list:
    print(f"\n{h}")
    param_gen = partial(generate_parameters, num_assets = 40, time_horizon = 200, min_hold = h)

    seeds = [100]

    df = experiment(seeds, param_gen = param_gen)

    runtime.append(df["runtime"].to_numpy())
    lp_gap.append(df["lp_gap"].to_numpy())

runtime = np.array(runtime).T
lp_gap = np.array(lp_gap).T

for row_idx, runtimes in enumerate(runtime):
    plt.plot(h_list, runtimes, marker="o",
             label=f"Row {row_idx}")

plt.xlabel("Holding Time (h)")
plt.xlim(0, max(h_list))
plt.ylabel("Runtime")
plt.legend()
plt.show()

for row_idx, gap in enumerate(lp_gap):
    plt.plot(h_list, gap, marker="o",
             label=f"Row {row_idx}")

plt.xlabel("Holding Time (h)")
plt.xlim(0, max(h_list))
plt.ylabel("Runtime")
plt.legend()
plt.show()



from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
from optimisation import det_ef_opt, Parameters

# #default parameters for input
# @dataclass
# class Parameters:
#     # Number of assets and time horizon
#     num_assets: int = 40
#     time_horizon: int = 200
#     seed_number: int = 100

#     # CAPEX constraint
#     C_CAPEX: float = 500.0

#     #Version of the optimization model to use
#     bpp: bool = False
#     vb: bool = False
#     tighterM: bool = False

#     # Defined with __post_init__
#     b: np.ndarray = field(init=False)
#     cost: np.ndarray = field(init=False)
#     L: np.ndarray = field(init=False)
#     U: np.ndarray = field(init=False)
#     holding_time: np.ndarray = field(init=False)
#     x_initial: np.ndarray = field(init=False)
#     def __post_init__(self):
#         #Minimum holding
#         self.holding_time = np.ones(self.num_assets) * 5
    
#         #initial positions
#         self.x_initial = np.zeros(self.num_assets)  # Initialize with zeros or any other desired initial positions
    
#         #Randomly generated expected returns and costs for each asset at each time period
#         rng = np.random.default_rng(self.seed_number)
#         self.b = rng.uniform(-10, 10, size=(self.num_assets, self.time_horizon))
#         self.cost = rng.uniform(0, 10, size=(self.num_assets, self.time_horizon))
    
#         #Upper/Lower bound
#         l = rng.uniform(0, 0.5, size=(self.num_assets, self.time_horizon))
#         u = rng.uniform(0.5, 1.0, size=(self.num_assets, self.time_horizon))
    
#         self.U = u
#         self.L = l

#this document will be for writing the code for the proper testing of optimisation problem
p = Parameters()


p_bp = Parameters()
p_bp = replace(p, bpp = True)
p_vb = replace(p, vb = True)
p_tm = replace(p, tighterM = True)
p_bpvb = replace(p_bp, vb = True)
p_vbtm = replace(p_vb, tighterM = True)
p_bpvbtm = replace(p_bpvb, tighterM = True)

#Experiment code
def experiment(test_classes, seeds, optimiser):
    results = []

    total = len(seeds) * len(test_classes)

    for it, seed in enumerate(seeds):

        for i, p in enumerate(test_classes):
            #Code start
            p.seed_number = seed

            result = optimiser(p)

            obj = result["objective_value"]
            lp_bound = result["lp_bound"]

            # Calculate LP integrality gap
            if (
                obj is not None
                and lp_bound is not None
                and np.isfinite(obj)
                and abs(obj) > 1e-10
            ):
                gap = abs(obj - lp_bound) / abs(obj) * 100
            else:
                gap = np.nan

            results.append({
                "test_class": i,
                "seed": seed,
                "objective": obj,
                "runtime": result["runtime"],
                "lp_bound": lp_bound,
                "lp_gap": gap
            })

            #Code progress output check how much is done
            progress = it * len(test_classes) + i + 1
            print(f"\rProgress: {progress}/{total} ({progress/total:.1%})", end="", flush=True)

    return pd.DataFrame(results)

all_classes = [p, p_bp, p_vb, p_tm, p_bpvb, p_vbtm, p_bpvbtm]
seeds = [100]

df = experiment(all_classes, seeds, det_ef_opt)
print(df)
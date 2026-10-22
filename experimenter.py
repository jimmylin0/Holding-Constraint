import numpy as np
import pandas as pd
from itertools import product
from optimisation import det_ef_opt, generate_parameters

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

#Experiment code
def experiment(seeds: list, optimiser = det_ef_opt):
    """
    This function runs an experiment using the optimiser for given list of seeds and all combination of parameter settings

    Input:
    seeds (list): List of seeds to be used to create the problem.
    optimiser (function): Optimisation function for the investment planning problem.

    Output:
    results (pd.DataFrame): Dataframe containing the results,
        Columns:
            bound_preprocess (bool): Bound preprocessing procedure.
            tighter_M (bool): Big M constraint tightening.
            variable_bound (bool): Variable upper/lower bounds.
            seed (int): Seed used to create problem instance.
            objective (float): Objective function of the solution.
            runtime (float): Runtime of the optimisation problem.
            lp_bound (float): Objective function of the LP relaxation.
            lp_gap (float): Gap between relaxation and optimal.
    """
    results = []

    #Enumerate possible parameter settings
    test_combinations = list(product([False, True], repeat = 3))

    total = len(seeds) * len(test_combinations)

    for it, seed in enumerate(seeds):
        #Generate parameters
        p = generate_parameters(seed)

        for i, (a, b, c) in enumerate(test_combinations):
            #Code start
            p.seed_number = seed

            result = optimiser(p, a, b, c)

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
                "bound_preprocess": a,
                "tighter_M": b,
                "variable_bound": c,
                "seed": seed,
                "objective": obj,
                "runtime": result["runtime"],
                "lp_bound": lp_bound,
                "lp_gap": gap
            })

            #Code progress output check how much is done
            progress = it * len(test_combinations) + i + 1
            print(f"\rProgress: {progress}/{total} ({progress/total:.1%})", end="", flush=True)

    return pd.DataFrame(results)

seeds = [100]

df = experiment(seeds)
print(f"\n{df}")
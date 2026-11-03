from collections.abc import Callable
import numpy as np
import pandas as pd
from itertools import product
from gurobipy import GRB
from optimisation import det_ef_opt, generate_parameters, Parameters

#this document will be for writing the code for the proper testing of optimisation problem

ParamGen = Callable[[int], Parameters]

#Experiment code
def experiment(seeds: list, param_gen: ParamGen, optimiser = det_ef_opt):
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
        p = param_gen(seed_number = seed)

        for i, (a, b, c) in enumerate(test_combinations):
            #Code start
            result = optimiser(p, a, b, c)

            obj = result["objective_value"]
            lp_bound = result["lp_bound"]

            # Calculate LP integrality gap
            if result["status"] == GRB.OPTIMAL:
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
                "lp_gap": gap,
                "node_count": result["node_count"]
            })

            #Code progress output check how much is done
            progress = it * len(test_combinations) + i + 1
            print(f"\rProgress: {progress}/{total} ({progress/total:.1%})", end="", flush=True)

    return pd.DataFrame(results)

if __name__ == "__main__":
    from functools import partial
    
    param_gen = partial(generate_parameters, num_assets = 40, time_horizon = 50, min_hold = 5)
    
    seeds = [100]

    df = experiment(seeds, param_gen = param_gen)
    print(f"\n{df}")
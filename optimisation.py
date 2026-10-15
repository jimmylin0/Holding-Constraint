import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Deterministic Equivalent Formulation
def det_ef_opt(parameters):
    """
    This function performs deterministic equivalent optimization on a given scenario tree.

    Parameters:
    parameters (dataclass): A dataclass containing all the necessary parameters for the optimization, including:
        - b: A 2D array of shape (num_assets, time_horizon) representing the expected returns for each asset at each time period.
        - cost: A 2D array of shape (num_assets, time_horizon) representing the costs associated with each asset at each time period.
        - C_CAPEX: A scalar representing the capital expenditure constraint.
        - U: A 2D array of shape (num_assets, time_horizon) representing the upper bounds for each asset at each time period.
        - L: A 2D array of shape (num_assets, time_horizon) representing the lower bounds for each asset at each time period.
        - holding_time: A 1D array of shape (num_assets,) representing the minimum holding time for each asset.
        - x_initial: A 1D array of shape (num_assets,) representing the initial positions for each asset.
        - version: A string indicating the version of the optimization model to use (default is "v1").

    Returns:
    dict: A dictionary containing the optimization results, including:
        - objective_value: The optimal value of the objective function.
        - runtime: The time taken to solve the optimization problem.
        - decision_variables: A dictionary of decision variable names and their optimal values.
    """

    time_horizon = parameters.b.shape[1]  # b is a 2D array with shape (num_assets, time_horizon)
    num_assets = len(parameters.b)  # Number of assets

    m = gp.Model("Deterministic_Equivalent_Optimization")

    #Create decision variables for each asset, time period, and scenario
    x = m.addVars(num_assets, time_horizon, vtype = GRB.CONTINUOUS, name = "x")

    #Objective function
    m.setObjective(gp.quicksum(parameters.b[j, t] * x[j, t] for j in range(num_assets) for t in range(time_horizon)), GRB.MAXIMIZE)

    #Constraints
    #CAPEX Constraint
    for t in range(time_horizon):
        m.addConstr(
            gp.quicksum(parameters.cost[j, t] * x[j, t] for j in range(num_assets)) <= parameters.C_CAPEX, name=f"CAPEX_constraint_{t}")

    #Auxiliary Binary Variables for Investment and Divestment
    delta = m.addVars(num_assets, time_horizon, vtype=GRB.BINARY, name="delta")

    #Big-M Constraints for Investment and Divestment
    if parameters.version == "1":
        #Upper and Lower Bounds for Decision Variables
        for j in range(num_assets):
            for t in range(time_horizon):
                x[j, t].lb = parameters.L[j, t]
                x[j, t].ub = parameters.U[j, t]

        for j in range(num_assets):
            #Use initial positions for the first time period to set the big-M constraints
            m.addConstr(x[j, 0] - parameters.x_initial[j] <= (parameters.U[j, 0] - parameters.x_initial[j]) * delta[j, 0], name=f"bigM_invest_{j}_0")
            m.addConstr(parameters.x_initial[j] - x[j, 0] <= (parameters.x_initial[j] - parameters.L[j, 0]) * delta[j, 0], name=f"bigM_divest_{j}_0")


            for t in range(1, time_horizon):
                m.addConstr(x[j, t] - x[j, t - 1] <= (parameters.U[j, t] - parameters.L[j, t - 1]) * delta[j, t], name=f"bigM_invest_{j}_{t}")
                m.addConstr(x[j, t - 1] - x[j, t] <= (parameters.U[j, t - 1] - parameters.L[j, t]) * delta[j, t], name=f"bigM_divest_{j}_{t}")
    
    #Version 2 tighter
    elif parameters.version == "2":
        #Temp U and L
        temp_U = np.column_stack((parameters.x_initial, parameters.U))
        temp_L = np.column_stack((parameters.x_initial, parameters.L))

        #Defining U_tilde and L_tilde for the tighter big-M constraints
        U_tilde = np.full(parameters.U.shape, np.nan)
        L_tilde = np.full(parameters.U.shape, np.nan)

        #Defining U_bar and L_bar for the tighter big-M constraints
        U_bar = np.full(parameters.U.shape, np.nan)
        L_bar = np.full(parameters.U.shape, np.nan)

        #Define U_tilde, L_tilde, U_bar, and L_bar based on the holding time for each asset
        for j in range(num_assets):
            for t in range(time_horizon):
                U_tilde[j, t] = temp_U[j, t + 1:min(time_horizon + 2, t + 2 + int(parameters.holding_time[j]))].min()
                L_tilde[j, t] = temp_L[j, t + 1:min(time_horizon + 2, t + 2 + int(parameters.holding_time[j]))].max()

                if t == 0:
                    U_bar[j, t] = parameters.x_initial[j]
                    L_bar[j, t] = parameters.x_initial[j]
                else:
                    U_bar[j, t] = temp_U[j, max(0, t + 1 - int(parameters.holding_time[j])):t + 1].min()
                    L_bar[j, t] = temp_L[j, max(0, t + 1 - int(parameters.holding_time[j])):t + 1].max()

        #Tighter Big-M Constraints for Investment and Divestment
        for j in range(num_assets):
            m.addConstr(x[j, 0] - parameters.x_initial[j] <= (U_tilde[j, 0] - parameters.x_initial[j]) * delta[j, 0], name=f"bigM_invest_{j}_0")
            m.addConstr(parameters.x_initial[j] - x[j, 0] <= (parameters.x_initial[j] - L_tilde[j, 0]) * delta[j, 0], name=f"bigM_divest_{j}_0")
        
            for t in range(1, time_horizon):
                m.addConstr(x[j, t] - x[j, t - 1] <= (U_tilde[j, t] - L_bar[j, t]) * delta[j, t], name=f"bigM_invest_{j}_{t}")
                m.addConstr(x[j, t - 1] - x[j, t] <= (U_bar[j, t] - L_tilde[j, t]) * delta[j, t], name=f"bigM_divest_{j}_{t}")

        #Tighter Upper and Lower Bounds for Decision Variables
        """double check later if correct logic for the upper and lower bounds"""
        for j in range(num_assets):
            for t in range(time_horizon):
                #Upper Bound Constraint
                m.addConstr(x[j, t] <= U_bar[j, t] 
                            + gp.quicksum((U_tilde[j, s] - U_bar[j, t])*delta[j, s] for s in range(max(0, t - int(parameters.holding_time[j])), t + 1))
                            , name=f"upper_bound_{j}_{t}")

                m.addConstr(x[j, t] <= U_tilde[j, t] 
                            + gp.quicksum((U_bar[j, s] - U_tilde[j, t])*delta[j, s] for s in range(t + 1, min(time_horizon, t + int(parameters.holding_time[j]) + 1)))
                            , name=f"upper_bound_2_{j}_{t}")
                
                #Lower Bound Constraint
                m.addConstr(x[j, t] >= L_bar[j, t]
                            + gp.quicksum((L_tilde[j, s] - L_bar[j, t])*delta[j, s] for s in range(max(0, t - int(parameters.holding_time[j])), t + 1))
                            , name=f"lower_bound_{j}_{t}")

                m.addConstr(x[j, t] >= L_tilde[j, t]
                            + gp.quicksum((L_bar[j, s] - L_tilde[j, t])*delta[j, s] for s in range(t + 1, min(time_horizon, t + int(parameters.holding_time[j]) + 1)))
                            , name=f"lower_bound_2_{j}_{t}")

    #Asset Holding Constraints: Ensure that if an asset is invested/divested, action cannot be taken until h years have passed.
    for j in range(num_assets):
        for t in range(time_horizon - int(parameters.holding_time[j])):
            m.addConstr(
                gp.quicksum(
                    delta[j, s] for s in range(t, t + int(parameters.holding_time[j]) + 1)
                ) <= 1,
                name=f"holding_constraint_{j}_{t}"
            )

    #Optimize the model
    m.optimize()

    #Check if the optimization was successful (edit later to return more information)
    if m.status == GRB.OPTIMAL:
        solution = {
            "objective_value": m.objVal,
            "runtime": m.Runtime,
            "decision_variables": {var.varName: var.x for var in m.getVars()},
        }

        return solution
    else:
        return {"status": m.status}

def bound_preprocess(parameters):
    """
    This function preprocesses the upper and lower bounds for each asset at each time period.

    Parameters:
    parameters (dataclass): A dataclass containing all the necessary parameters for the optimization, including:
        - U: A 2D array of shape (num_assets, time_horizon) representing the upper bounds for each asset at each time period.
        - L: A 2D array of shape (num_assets, time_horizon) representing the lower bounds for each asset at each time period.
        - holding_time: A 1D array of shape (num_assets,) representing the minimum holding time for each asset.
    """

    u = parameters.U.copy()
    l = parameters.L.copy()
    h = parameters.holding_time

    time_horizon = parameters.U.shape[1]  # U is a 2D array with shape (num_assets, time_horizon)
    num_assets = len(parameters.U)

    for j in range(num_assets):
        for t in range(1, time_horizon):
            for s in range(1, int(min(h[j], time_horizon - t))):
                if u[j, t] > max(u[j, t-1], u[j, t+s]):
                    u[j, t] = max(u[j, t-1], u[j, t+s])
                if l[j, t] < min(l[j, t-1], l[j, t+s]):
                    l[j, t] = min(l[j, t-1], l[j, t+s])

    return u, l

def plot_bounds(u, l, asset_idx=0):
    """
    Plot upper and lower bounds for a single asset.

    Parameters
    ----------
    u: A 2D array upper bounds with shape (num_assets, time_horizon).
    l: A 2D array lower bounds with shape (num_assets, time_horizon).
    asset_idx: int index of the asset to plot.

    Returns
    -------
    fig, ax: Matplotlib figure and axis objects.
    """
    if u.shape != l.shape:
        raise ValueError("u and l must have the same shape.")

    if u.ndim != 2:
        raise ValueError("u and l must be 2D arrays.")

    if not 0 <= asset_idx < u.shape[0]:
        raise IndexError("asset_idx is out of range.")

    x = np.arange(u.shape[1])

    upper = u[asset_idx]
    lower = l[asset_idx]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, lower, label="Lower bound")
    ax.plot(x, upper, label="Upper bound")
    ax.fill_between(x, lower, upper, alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"Upper and Lower Bounds — Asset {asset_idx}")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig, ax

###debug delete after fixed
def compare_solutions(result, result_preprocessed, tol=1e-6):
    x1 = result["decision_variables"]
    x2 = result_preprocessed["decision_variables"]

    # Align variables by name, including variables missing from either model
    df = pd.DataFrame({
        "original": pd.Series(x1, dtype=float),
        "preprocessed": pd.Series(x2, dtype=float),
    })

    df["difference"] = df["original"] - df["preprocessed"]
    df["abs_difference"] = df["difference"].abs()

    # Identify missing variables separately
    df["missing_in_original"] = df["original"].isna()
    df["missing_in_preprocessed"] = df["preprocessed"].isna()

    # Only compare numerical values where both exist
    df["different"] = (
        (df["abs_difference"] > tol)
        | df["missing_in_original"]
        | df["missing_in_preprocessed"]
    )

    differences = df[df["different"]].sort_values(
        "abs_difference", ascending=False, na_position="last"
    )

    print("Objective values:")
    print(f"Original:     {result['objective_value']}")
    print(f"Preprocessed: {result_preprocessed['objective_value']}")

    print(f"\nTotal variables (union): {len(df)}")
    print(f"Different variables:     {len(differences)}")
    print(f"Missing in original:     {df['missing_in_original'].sum()}")
    print(f"Missing in preprocessed: {df['missing_in_preprocessed'].sum()}")
    print(f"Max absolute difference: {df['abs_difference'].max()}")

    return differences, df
#################

if __name__ == "__main__":
    from dataclasses import dataclass, field
    import time

    #parameters for input
    @dataclass
    class Parameters:
        # Number of assets and time horizon
        num_assets: int = 40
        time_horizon: int = 200
        seed_number: int = 100

        # CAPEX constraint
        C_CAPEX: float = 500.0

        #Version of the optimization model to use
        version: str = "1"  # Default version is "1"

        # Defined with __post_init__
        b: np.ndarray = field(init=False)
        cost: np.ndarray = field(init=False)
        L: np.ndarray = field(init=False)
        U: np.ndarray = field(init=False)
        holding_time: np.ndarray = field(init=False)
        x_initial: np.ndarray = field(init=False) 

        def __post_init__(self):
            #Minimum holding
            self.holding_time = np.ones(self.num_assets) * 25

            #initial positions
            self.x_initial = np.zeros(self.num_assets)  # Initialize with zeros or any other desired initial positions

            #Randomly generated expected returns and costs for each asset at each time period
            rng = np.random.default_rng(self.seed_number)
            self.b = rng.uniform(-10, 10, size=(self.num_assets, self.time_horizon))
            self.cost = rng.uniform(0, 10, size=(self.num_assets, self.time_horizon))

            #
            #Upper/Lower bound
            l = rng.uniform(0, 0.5, size=(self.num_assets, self.time_horizon))
            u = rng.uniform(0.5, 1.0, size=(self.num_assets, self.time_horizon))

            self.U = u
            self.L = l

    #run optimisation and track runtime
    
    full_runtime_timer_start = time.time()
    result = det_ef_opt(Parameters())
    full_runtime_timer_end = time.time()

    model_building_time = full_runtime_timer_end - full_runtime_timer_start - result['runtime']

    #For tracking objective function value
    print(f"Objective Value: {result['objective_value']}")
    print(f"Runtime: {result['runtime']} seconds")
    print(f"Model Building Time: {model_building_time} seconds")
    """
    #plot bounds for the first asset without preprocessing
    p = Parameters()
    fig, ax = plot_bounds(p.U, p.L, asset_idx=0)
    plt.show()
    """
    #run optimisation with the bound preprocess procedure and track runtime
    u, l = bound_preprocess(Parameters())
    p = Parameters()
    p.U = u
    p.L = l
    p.version = "2"  # Use version 2 for the optimization model

    #plot bounds for the first asset with preprocessing
    #fig, ax = plot_bounds(p.U, p.L, asset_idx=0)
    #plt.show()
    
    full_runtime_timer_start2 = time.time()
    result_preprocessed = det_ef_opt(p)
    full_runtime_timer_end2 = time.time()
    
    model_building_time = full_runtime_timer_end2 - full_runtime_timer_start2 - result_preprocessed['runtime']
    
    #For tracking objective function value
    print(f"Objective Value: {result_preprocessed['objective_value']}")
    print(f"Runtime: {result_preprocessed['runtime']} seconds")
    print(f"Model Building Time: {model_building_time} seconds")

    #debug in case to compare the optimisation output with and without the bound preprocess procedure
    #if correct difference should be zero and no missing variables
    #differences, comparison = compare_solutions(
    #    result,
    #    result_preprocessed
    #)

    #print(differences.to_string())
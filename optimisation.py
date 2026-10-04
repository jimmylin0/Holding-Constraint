import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
    x = m.addVars(num_assets, time_horizon, vtype = GRB.CONTINUOUS, name = "x", lb = 0, ub = 1)
    #x_invest = m.addVars(num_assets, time_horizon - 1, vtype = GRB.CONTINUOUS, name = "x_invest", lb = 0, ub = 1)
    #x_divest = m.addVars(num_assets, time_horizon - 1, vtype = GRB.CONTINUOUS, name = "x_divest", lb = 0, ub = 1)

    epsilon = 1e-6  # Small value to avoid degeneracy

    #Objective function
    m.setObjective(gp.quicksum(parameters.b[j, t] * x[j, t] for j in range(num_assets) for t in range(time_horizon)), GRB.MAXIMIZE) #avoid degeneracy

    #Constraints

    #Portfolio Flow Conservation:
    #for j in range(num_assets):
    #    for t in range(time_horizon - 1):
    #        m.addConstr(x[j, t + 1] == x[j, t] + x_invest[j, t] - x_divest[j, t], name=f"flow_conservation_{j}_{t}")

    #CAPEX Constraint
    for t in range(time_horizon):
        m.addConstr(
            gp.quicksum(parameters.cost[j, t] * x[j, t] for j in range(num_assets)) <= parameters.C_CAPEX, name=f"CAPEX_constraint_{t}")

    #Auxiliary Binary Variables for Investment and Divestment
    delta = m.addVars(num_assets, time_horizon - 1, vtype=GRB.BINARY, name="delta")

    #Big-M Constraints for Investment and Divestment
    for j in range(num_assets):
        #Use initial positions for the first time period to set the big-M constraints
        m.addConstr(x[j, 1] - parameters.x_initial[j] <= max(0, parameters.U[j, 1] - parameters.x_initial[j]) * delta[j, 0], name=f"bigM_invest_{j}_0")
        m.addConstr(parameters.x_initial[j] - x[j, 1] <= max(0, parameters.U[j, 0] - parameters.L[j, 1]) * delta[j, 0], name=f"bigM_divest_{j}_0")

        for t in range(1, time_horizon - 1):
            m.addConstr(x[j, t + 1] - x[j, t] <= max(0, parameters.U[j, t + 1] - parameters.L[j, t]) * delta[j, t], name=f"bigM_invest_{j}_{t}")
            m.addConstr(x[j, t] - x[j, t + 1] <= max(0, parameters.U[j, t] - parameters.L[j, t + 1]) * delta[j, t], name=f"bigM_divest_{j}_{t}")

    #Asset Holding Constraints: Ensure that if an asset is invested/divested, action cannot be taken until h years have passed.
    for j in range(num_assets):
        for t in range(time_horizon - int(parameters.holding_time[j])):
            m.addConstr(
                gp.quicksum(
                    delta[j, s] for s in range(t, t + int(parameters.holding_time[j]))
                ) <= 1,
                name=f"holding_constraint_{j}_{t}"
            )


    #Upper and Lower Bounds for Decision Variables
    for j in range(num_assets):
        for t in range(1, time_horizon):
            m.addConstr(x[j, t] >= parameters.L[j, t], name=f"lower_bound_{j}_{t}")
            m.addConstr(x[j, t] <= parameters.U[j, t], name=f"upper_bound_{j}_{t}")

    #Initial Positions x
    for j in range(num_assets):
        m.addConstr(x[j, 0] == parameters.x_initial[j], name=f"initial_position_{j}")

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


if __name__ == "__main__":
    from dataclasses import dataclass, field
    import time

    #parameters for input
    @dataclass
    class Parameters:
        # Number of assets and time horizon
        num_assets: int = 20
        time_horizon: int = 100
        seed_number: int = 100

        # CAPEX constraint
        C_CAPEX: float = 50.0

        # Defined with __post_init__
        b: np.ndarray = field(init=False)
        cost: np.ndarray = field(init=False)
        L: np.ndarray = field(init=False)
        U: np.ndarray = field(init=False)
        holding_time: np.ndarray = field(init=False)
        x_initial: np.ndarray = field(init=False) 

        def __post_init__(self):
            #Minimum holding
            self.holding_time = np.ones(self.num_assets) * 2

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

            l[:, 0] = self.x_initial
            u[:, 0] = self.x_initial

            self.U = u
            self.L = l

    full_runtime_timer_start = time.time()
    result = det_ef_opt(Parameters())
    full_runtime_timer_end = time.time()

    model_building_time = full_runtime_timer_end - full_runtime_timer_start - result['runtime']

    #For tracking objective function value
    print(f"Objective Value: {result['objective_value']}")
    print(f"Runtime: {result['runtime']} seconds")
    print(f"Model Building Time: {model_building_time} seconds")

    u, l = bound_preprocess(Parameters())
    p = Parameters()
    p.U = u
    p.L = l

    full_runtime_timer_start2 = time.time()
    result_preprocessed = det_ef_opt(p)
    full_runtime_timer_end2 = time.time()
    
    model_building_time = full_runtime_timer_end2 - full_runtime_timer_start2 - result_preprocessed['runtime']
    
    #For tracking objective function value
    print(f"Objective Value: {result_preprocessed['objective_value']}")
    print(f"Runtime: {result_preprocessed['runtime']} seconds")
    print(f"Model Building Time: {model_building_time} seconds")
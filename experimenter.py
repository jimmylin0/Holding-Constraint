from optimisation import det_ef_opt, bound_preprocess
from dataclasses import dataclass, field, replace
import time
import numpy as np

#default parameters for input
@dataclass
class Parameters:
    # Number of assets and time horizon
    num_assets: int = 40
    time_horizon: int = 400
    seed_number: int = 100

    # CAPEX constraint
    C_CAPEX: float = 500.0

    #Version of the optimization model to use
    vb: bool = False
    tighterM: bool = False

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
    
        #Upper/Lower bound
        l = rng.uniform(0, 0.5, size=(self.num_assets, self.time_horizon))
        u = rng.uniform(0.5, 1.0, size=(self.num_assets, self.time_horizon))
    
        self.U = u
        self.L = l

#this document will be for writing the code for the proper testing of optimisation problem
p = Parameters()

new_u, new_l = bound_preprocess(p)
p_bp = Parameters()
p_bp.U = new_u
p_bp.L = new_l
p_vb = replace(p, vb = True)
p_tm = replace(p, tighterM = True)
p_bpvb = replace(p_bp, vb = True)
p_vbtm = replace(p_vb, tighterM = True)
p_bpvbtm = replace(p_bpvb, tighterM = True)

print(p_bpvbtm)
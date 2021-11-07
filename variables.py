import numpy as np

def init_variables(lambda_stake: float, lambda_collateral: float, lambda_borrow: float, lambda_slash: float, size: int, T_max: int):
    stakes = np.zeros(((T_max, size)))
    stakes[0] = np.random.exponential(lambda_stake, size)
    collateral = np.random.beta(1, lambda_collateral, size)
    borrow = np.random.beta(1, lambda_borrow, size)
    slash = np.random.beta(1, lambda_slash, size)
    loans = np.zeros((T_max, size), dtype=float)
    phi = np.zeros((T_max, size), dtype=float)
    return stakes, collateral, borrow, slash, loans, phi

def from_demand_to_borrowing_array(borrow: np.ndarray, T_max: int) -> np.ndarray:
    return np.random.binomial(1, borrow, (T_max, len(borrow)))
    
def from_slashing_probability_to_slashing_array(slash: np.ndarray, T_max: int) -> np.ndarray:
    return np.random.binomial(1, slash, (T_max, len(slash)))

def getBlockReward(T_max: int) -> np.ndarray:
    init = np.ones(T_max)
    for i in range(10):
        init[i * T_max // 10: min((i + 1) * T_max // 10, T_max)] = 11 - i
    return init

def init_stakes_issued(size: int):
    return np.zeros(size)
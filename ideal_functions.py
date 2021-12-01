import numpy as np
from variables import *

k = 2
phi_max = 2

def mother_pricing_curve(x: np.ndarray) -> np.ndarray:
    global k
    res = 1 / (x ** k)
    res[x < 1] = 1
    return res
    
def update_borrowers(l: np.ndarray, X: np.ndarray, c: np.ndarray, stakes: np.ndarray, h: int, T_max: int) -> np.ndarray:
    """
    l: loans array (T_max * n)
    X: borrowing array (T_max * n)
    c: collateral factor array (1 * n)
    stakes: array of stakes (T_max * n)
    h: current period
    """
    # Will they borrow?
    if h == T_max - 1:
        raise Exception("Last period, no borrowing")
    borrowers: np.ndarray = np.logical_and(X[h] == 1, l[h] < c * stakes[h])
    n_borrowers = borrowers.sum()
    eps = np.random.uniform(0, 1, n_borrowers)
    borrow_amt_as_perc_of_stake = eps * (c[borrowers] - l[h][borrowers] / stakes[h][borrowers])
    l[h + 1][borrowers] = l[h][borrowers] + borrow_amt_as_perc_of_stake * stakes[h][borrowers]
    l[h + 1][~borrowers] = l[h][~borrowers]
    return l

def mark_loans_at_current_height(c: np.ndarray, stakes: np.ndarray, l: np.ndarray, stakes_issued: np.ndarray, h: int) -> np.ndarray:
    """
    c: collateral factor array (1 * n)
    stakes: array of stakes (T_max * n)
    l: loans array (T_max * n)
    stakes_issued: array of stakes issued (1 * n)
    h: current period
    """
    phi = np.ones(len(c))
    borrowers = l[h] > 0
    b = c / (1 - c)
    a = 1 / (stakes_issued * (c - 1))
    phi[borrowers] = mother_pricing_curve(a[borrowers] * stakes[h][borrowers] + b[borrowers])
    return phi

def clean_defaulted_loans(phi: np.ndarray, stakes: np.ndarray, borrowing_array: np.ndarray, h: int):
    """
    phi: array of phi values (T_max * n)
    stakes: array of stakes (T_max * n)
    borrowing_array: array of borrowing (T_max * n)
    h: current period
    """
    if h == len(stakes) - 1:
        raise Exception("Last period, no defaulting")
    global phi_max
    defaulted = phi[h] > phi_max
    stakes[h + 1][defaulted] = 0
    stakes[h + 1][~defaulted] = stakes[h][~defaulted]
    # borrowing_array[h + 1:][defaulted] = 0
    for i in np.arange(len(borrowing_array[h]))[defaulted]:
        borrowing_array[h + 1:][i] = 0
    return stakes, borrowing_array

def update_stake_distribution(stakes: np.ndarray, slashing_array: np.ndarray, h: int, iota: float, reward: float) -> np.ndarray:
    """
    stakes: array of stakes (T_max * n)
    slashing_array: array of slashing (T_max * n)
    h: current period
    iota: interest rate
    reward: reward rate
    """
    if h == len(stakes) - 1:
        raise Exception("Last period, no slashing")
    # choose validator proportional to stake
    validator_index = np.random.choice(np.arange(len(stakes[h])), p=stakes[h] / np.sum(stakes[h]))
    validator = np.zeros(len(stakes[h]), dtype=bool)
    validator[validator_index] = True
    # nor validator nor slashed
    stakes[h + 1] = stakes[h]
    stakes[h + 1][validator_index] = stakes[h][validator_index] + reward[h]
    slashed = slashing_array[h] == 1
    stakes[h + 1][slashed] = stakes[h][slashed] * (1 - iota)
    return stakes

def loop(lambda_stake, lambda_collateral, lambda_borrow, lambda_slash, size, T_max, iota, epoch_time):
    stakes, collateral, borrow, slash, loans, phi = init_variables(lambda_stake, lambda_collateral, lambda_borrow, lambda_slash, size, T_max)
    borrowing_array = from_demand_to_borrowing_array(borrow, T_max)
    slashing_array = from_slashing_probability_to_slashing_array(slash, T_max)
    stakes_issued = init_stakes_issued(size)
    R = getBlockReward(T_max)
    for h in range(T_max - 1):
        if h % epoch_time == 0:
            loans = update_borrowers(loans, borrowing_array, collateral, stakes, h, T_max)
            stakes_issued = stakes[h]
        current_phi = mark_loans_at_current_height(collateral, stakes, loans, stakes_issued, h)
        phi[h] = current_phi
        stakes, borrowing_array = clean_defaulted_loans(phi, stakes, borrowing_array, h)
        stakes = update_stake_distribution(stakes, slashing_array, h, iota, R)
    return stakes, loans, borrowing_array, phi

def main():
    lambda_stake = 50
    lambda_collateral = .75
    lambda_borrow = 1
    lambda_slash = 1
    size = 100
    T_max = 1000
    iota = 0.1
    epoch_time = 1
    
    stakes, loans, borrowing_array, phi = loop(lambda_stake, lambda_collateral, lambda_borrow, lambda_slash, size, T_max, iota, epoch_time)

if __name__ == "__main__":
    main()
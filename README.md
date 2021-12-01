# POS Agent simulation

&copy; Maxence Raballand 2021

This proof of stake validation mechanism simulation will try to show inequalities in stakes with the Gini coefficient. We assume that a staking derivatives exists based on Tarun Chitra Paper: "Why stake when you can borrow ?".

The pseudo-code of the functions can be found on the original white paper from Tarun Chitra : ["Why stake when you can borrow ?"](https://arxiv.org/abs/2006.11156).

## Variable initialization

First, we will define some constants from [this file](variables.py).

```py
def init_variables(lambda_stake: float, lambda_collateral: float, lambda_borrow: float, lambda_slash: float, size: int, T_max: int):
    stakes = np.zeros(((T_max, size)))
    stakes[0] = np.random.exponential(lambda_stake, size)
    collateral = np.random.beta(1, lambda_collateral, size)
    borrow = np.random.beta(1, lambda_borrow, size)
    slash = np.random.beta(1, lambda_slash, size)
    loans = np.zeros((T_max, size), dtype=float)
    phi = np.zeros((T_max, size), dtype=float)
    return stakes, collateral, borrow, slash, loans, phi
```

This function takes some parameters that are determined form the paper or will serve as a variable to test different hypothesis during simulations. As we can see here, the starting stakes are following an exponential law because the distribution matches the reality. The parameter of that law will be the mean of the stakes of the protocol we will put to the test.

For the collateral factor, the borrowing demand, and the slashing probabilities, we have beta lews with parameter of 1 and a variable that we will test during simulations.

Finally, we initialize the rest of the variables as empty matrixes. In the rest of the code, we will also transform the borrowing and slashing probability to get an array of 0 and 1 corresponding to whether the user will borrow or not and get slashed or not at a given period.

```py
def from_demand_to_borrowing_array(borrow: np.ndarray, T_max: int) -> np.ndarray:
    return np.random.binomial(1, borrow, (T_max, len(borrow)))
    
def from_slashing_probability_to_slashing_array(slash: np.ndarray, T_max: int) -> np.ndarray:
    return np.random.binomial(1, slash, (T_max, len(slash)))
```

In this file, I also define a reward function that takes into account the block height in order to create a halving effect. This reward function can then be passed as a variable so that I can create other ones in order to simulate other protocols.

```py
def getBlockReward(T_max: int) -> np.ndarray:
    init = np.ones(T_max)
    for i in range(10):
        init[i * T_max // 10: min((i + 1) * T_max // 10, T_max)] = 110 - i * 10
    return init
```

## Ideal Functions implementation

In this [first implementation](ideal_functions.py), there are simple functions that describe the different steps we can in the loop, thus representing a block.

In our simulation, we will have four main functions. The first function `update_borrowers` will make the user borrow or not based on the array of zeros and ones we created earlier and whether they can still borrow or not based on their collateral factor.

```py
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
```

Then, we have to mark the loans of these borrowers. But first, we have to create the mother pricing curve to determine how much a user as to pay in order to recover its loans. You can find more details on the pricing functions in the paper.

```py
k = 2

def mother_pricing_curve(x: np.ndarray) -> np.ndarray:
    global k
    res = 1 / (x ** k)
    res[x < 1] = 1
    return res
```

After that, we can finally mark the user loans at current height.

```py
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
```

Now that the loans are marked, we can clean defaulted loans from this period. A default is characterized by the fact that you have less money that the collateral factor times your stake when borring (passed in the pricing function). In other words, the result of the pricing function is greater than the phi_max.

```py
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
```

Finally, we have to act as the protocol as the underlying of this derivative is staking. So a staker is chosen randomly according to its share in the total amount staked, and gets the reward. From the slashing binomial array calculated before, we take all slashed user and remove *iota* % of their stake. If a validator is slashed, the reward is cancelled and the block is skipped.

```py
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
```

## Goal and results

The goal here is to determine the distribution of wealth. The better the distribution of wealth is, the better the protocol security is (in theory). To compute the distribution of wealth, we use the [Gini coefficient](https://en.wikipedia.org/wiki/Gini_coefficient). This indicator is used to compute the distribution of wealth in coutries for example. Here we simply pass the resulting stakes at the end of our simulation.

```py
def gini_coefficient(x: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a distribution.

    Parameters
    ----------
    x : np.ndarray
        Array of values.

    Returns
    -------
    float
        Gini coefficient.
    """
     # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
```

The distribution is calculated function of the borrowing demand and the slashing probability. The simulation is done a 100 times (law of big numbers) on periods of 1000 blocks(could be less). We end up with this heat map representing the Gini coefficient.

![Gini coefficient function of the borrowing demand and slashing probability](results/ideal_function_gini_heatmap.svg)

Here we have the same results as in the paper. The slashing probability does'nt really affect the Gini coefficient. However, if the borrowing demand is higher, the wealth distribution is getting better.

The difference with the paper is here with the value of the Gini coefficient that is supposed to be just above 0 when the borrowing demand is high.
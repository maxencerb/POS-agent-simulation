import numpy as np

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
    sum_abs_diff = 0
    for i in x:
        sum_abs_diff += np.abs(x - i).sum()
    return sum_abs_diff / (2 * x.size ** 2 * x.mean())
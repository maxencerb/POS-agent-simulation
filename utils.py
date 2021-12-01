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
     # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
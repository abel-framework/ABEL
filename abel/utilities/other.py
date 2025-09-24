# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
A collection of useful functions
"""

import numpy as np



###################################################
def find_closest_value_in_arr(arr, val):
    """
    Find the index and element in a 1D array ``arr``that is closest to a given 
    value ``val``.

    The function computes the absolute difference between each element of the 
    array and the target value, then returns the index of the element with the 
    smallest difference along with the element itself.

    Parameters
    ----------
    arr : float ndarray
        1D array of numeric values to search.

    val : float
        Target value to compare against.

    Returns
    -------
    idx : int
        Index of the element in ``arr`` that is closest to ``val``.

    closest_val : float
        The element in ``arr`` that is closest to ``val``.

    Examples
    --------
    >>> arr = np.array([1.0, 2.5, 4.0, 6.0])
    >>> find_closest_value_in_arr(arr, 3.2)
    (1, 2.5)

    Notes
    -----
    If there are multiple elements equally close to ``val``,
    the function returns the first one it encounters (lowest index).
    """
    idx = np.abs(arr - val).argmin()  # Find index of the closest value
    closest_val = arr[idx]  # Get the closest value
    
    return idx, closest_val



###################################################
def pad_downwards(val, padding=0.05):
    """
    Adjust a value ``val`` downward by a given percentage ``padding``.

    This function is typically used for adjusting axis limits or other bounds so 
    that the minimum value is padded "downwards" (further away from zero).

    Behavior:
    - If ``val > 0``: decreases val by ``padding`` percent.
        Example: ``val=100``, ``padding=0.1`` → 90.0
    - If ``val < 0``: makes ``val`` more negative by ``padding`` percent.
        Example: ``val=-50``, ``padding=0.1`` → -55.0
    - If ``val == 0``: returns 0 unchanged.
    - Special case: if ``padding == 1.0``
        - ``arr_min > 0`` → returns 0 (collapses completely to zero)
        - ``arr_min < 0`` → doubles in magnitude
        - ``arr_min == 0`` → returns 0

    Parameters
    ----------
    val : float
        The minimum value to adjust.

    padding : float, default=0.05
        Fraction (0–1 typically) by which to pad the value. Negative inputs are 
        converted to their absolute value.

    Returns
    -------
    float
        The adjusted minimum value.
    """
    if padding < 0.0:
        padding = np.abs(padding)
    return val*(1.0 - np.sign(val)*padding)



###################################################
def pad_upwards(val, padding=0.05):
    """
    Adjust a value ``val`` upward by a given percentage ``padding``.

    This function is typically used for adjusting axis limits or other bounds so 
    that the maximum value is padded "upwards" (further away from zero).

    Behavior
    --------
    - If ``val > 0``: increases ``val`` by ``padding`` percent.
        Example: ``val=100``, ``padding=0.1`` → 110.0
    - If ``val < 0``: makes ``val`` less negative by ``padding`` percent.
        Example: ``val=-50``, ``padding=0.1`` → -45.0
    - If ``val == 0``: returns 0 unchanged.
    - Special case: if ``padding == 1.0``
        - ``val > 0`` → doubles in magnitude
        - ``val < 0`` → collapses completely to 0
        - ``val == 0`` → returns 0

    Parameters
    ----------
    val : float
        The value to adjust.

    padding : float, default=0.05
        Fraction (0–1 typically) by which to pad the value.
        Negative inputs are converted to their absolute value.

    Returns
    -------
    float
        The adjusted value.
    """
    if padding < 0.0:
        padding = np.abs(padding)
    return val*(1.0 + np.sign(val)*padding)
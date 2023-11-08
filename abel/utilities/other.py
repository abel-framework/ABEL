"""
A collection of useful functions
"""

import numpy as np


def find_closest_value_in_arr(arr, val):
    """
    Find index and element in a 1D array that is closest to a given value.
    """
    idx = np.abs(arr - val).argmin()  # Find index of the closest value
    closest_val = arr[idx]  # Get the closest value
    
    return idx, closest_val
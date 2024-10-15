"""
A collection of useful functions
"""

import numpy as np



###################################################
def find_closest_value_in_arr(arr, val):
    """
    Find index and element in a 1D array that is closest to a given value.
    """
    idx = np.abs(arr - val).argmin()  # Find index of the closest value
    closest_val = arr[idx]  # Get the closest value
    
    return idx, closest_val



###################################################
def pad_downwards(arr_min, padding=0.05):
    """
    Reduces arr_min by the padding percentage if arr_min is positive, and increases it (more negative) by the padding percentage if arr_min is negative.
    """
    if padding < 0.0:
        padding = np.abs(padding)
    return arr_min*(1.0 - np.sign(arr_min)*padding)



###################################################
def pad_upwards(arr_max, padding=0.05):
    """
    Increases arr_max by the padding percentage if arr_max is positive, and decreases it (less negative) by the padding percentage if arr_max is negative.
    """
    if padding < 0.0:
        padding = np.abs(padding)
    return arr_max*(1.0 + np.sign(arr_max)*padding)
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import numpy as np

# remove the extreme percentiles
def clean_mask(values, enable=True, cut_percentile=5.0):
    """
    Generate a boolean mask that filters values based on percentile thresholds.

    Parameters
    ----------
    values : float array_like
        Input numeric array from which to compute percentile-based masking.

    enable : bool, optional
        Whether to apply percentile-based filtering. If ``False``, the function 
        returns a mask of all ``True`` values. Defaults to ``True``.

    cut_percentile : float, optional
        The lower and upper percentile cutoff. Values below the 
        ``cut_percentile`` percentile or above the ``100 - cut_percentile`` 
        percentile are masked out. Defaults to 5.0 (i.e., the mask can be used 
        to remove the lowest 5% and the highest 5% elements from ``values``).

    Returns
    -------
    bool ndarray
        Boolean array of the same shape as ``values``, where ``True`` indicates 
        that the value lies within the specified percentile range.

    Notes
    -----
    - If ``values`` is empty or ``enable`` is False, a mask of all ``True`` 
    values is returned, i.e. all values are considered to be inside the 
    percentile range.
    """
    if enable and values.size > 0:
        return np.logical_and(values > np.percentile(values, cut_percentile), values < np.percentile(values, 100-cut_percentile))
    else:
        return np.full(values.shape, True)


def prct_clean(x, enable=True, cut_percentile=5.0):
    """
    Filter an array by removing extreme values based on percentile thresholds.

    Parameters
    ----------
    x : float array_like
        Input numeric array to be filtered.

    enable : bool, optional
        Whether to perform percentile-based filtering. If ``False``, the 
        function returns the input array ``x`` unchanged. Defaults to ``True``.

    cut_percentile : float, optional
        The lower and upper percentile cutoff. Values below the 
        ``cut_percentile`` percentile or above the ``100 - cut_percentile`` 
        percentile are masked out. Defaults to 5.0.

    Returns
    -------
    float ndarray
        Filtered array containing only values within the specified percentile 
        range. If ``enable`` is ``False``, or if ``x`` is empty, the original 
        array is returned.

    Notes
    -----
    - Uses `clean_mask` to compute the mask for filtering.
    - By default, keeps values between the 5th and 95th percentiles
      (removes the extreme 10% of data).
    """
    #return x[clean_mask[x], enable=enable, cut_percentile=cut_percentile]
    mask = clean_mask(x, enable=enable, cut_percentile=cut_percentile)
    return x[mask]
    

def prct_clean2d(xs, ys, enable=True, cut_percentile=5.0):
    """
    Remove paired points that fall outside percentile thresholds on either 
    dimension.

    Parameters
    ----------
    xs : 1D float ndarray
        Input numeric array to be filtered.

    ys : 1D float ndarray
        Input numeric array to be filtered. Must be the same shape as `xs`.

    enable : bool, optional
        Whether to perform percentile-based filtering. If ``False``, the 
        function returns the input arrays ``x`` and ``y`` unchanged. Defaults to 
        ``True``.

    cut_percentile : float, optional
        The lower and upper percentile cutoff. Values below the 
        ``cut_percentile/2`` percentile or above the ``100 - cut_percentile/2`` 
        percentile are masked out if either the value in ``xs`` or ``ys`` is 
        outside the percentile range. Defaults to 5.0 (i.e., the function keeps 
        values within the middle 95% of each dimension).

    Returns
    -------
    tuple of numpy.ndarray
        A pair `(xs_filtered, ys_filtered)` where only points whose
        x- and y-values lie within the allowed percentile ranges are retained.

    Notes
    -----
    - A point is removed if either its x-value or y-value is an outlier.
    - For the default `cut_percentile = 5`, the function keeps values within
      the middle 95% of each dimension (2.5% trimmed from each tail).
    """
    mask_x = clean_mask(xs, enable=enable, cut_percentile=cut_percentile/2)
    mask_y = clean_mask(ys, enable=enable, cut_percentile=cut_percentile/2)
    mask = np.logical_and(mask_x, mask_y)
    return xs[mask], ys[mask]


def weighted_mean(values, weights, clean=False, cut_percentile=5.0):
    """
    Compute a weighted mean with optional percentile-based outlier removal.

    
    Parameters
    ----------
    values : float array_like
        Numeric values.

    weights : float array_like
        Weights corresponding to each value. Must be the same shape as 
        ``values``.

    clean : bool, optional
        Whether to apply percentile filtering to remove outliers before
        computing the weighted mean. Defaults to ``False``.

    cut_percentile : float, optional
        The lower and upper percentile cutoff. Values below the 
        ``cut_percentile`` percentile or above the ``100 - cut_percentile`` 
        percentile are masked out. Defaults to 5.0 (i.e., ignores the extreme 
        10% of data and only considers values between the 5th and 95th 
        percentiles.

    
    Returns
    -------
    float
        The weighted mean of ``values``.
    """

    if np.sum(weights) == 0.0:
        raise ZeroDivisionError("Weights sum to zero, cannot be normalized.")
    mask = clean_mask(values, enable=clean, cut_percentile=cut_percentile)
    return np.average(values[mask], weights=weights[mask])


def weighted_std(values, weights, clean=False, cut_percentile=5.0):
    """
    Compute a weighted standard deviation with optional percentile-based outlier 
    removal.

    
    Parameters
    ----------
    values : float array_like
        Numeric values.

    weights : float array_like
        Weights corresponding to each value. Must be the same shape as 
        ``values``.

    clean : bool, optional
        Whether to apply percentile filtering to remove outliers before
        computing the weighted mean. Defaults to ``False``.

    cut_percentile : float, optional
        The lower and upper percentile cutoff. Values below the 
        ``cut_percentile`` percentile or above the ``100 - cut_percentile`` 
        percentile are masked out. Defaults to 5.0 (i.e., ignores the extreme 
        10% of data and only considers values between the 5th and 95th 
        percentiles.

    
    Returns
    -------
    float
        The weighted standard deviation of ``values``.
    """
    if np.sum(weights) == 0.0:
        raise ZeroDivisionError("Weights sum to zero, cannot be normalized.")
    mask = clean_mask(values, enable=clean, cut_percentile=cut_percentile)
    return np.sqrt(np.cov(values[mask], aweights=weights[mask]))


def weighted_cov(xs, ys, weights, clean=False, cut_percentile=5.0):
    """
    Compute a weighted covariance between two arrays with optional 
    percentile-based outlier removal.

    
    Parameters
    ----------
    xs : 1D float ndarray
        Input numeric array.

    ys : 1D float ndarray
        Input numeric array. Must be the same shape as `xs`.

    weights : float array_like
        Weights corresponding to each value. Must be the same shape as 
        ``values``.

    clean : bool, optional
        Whether to apply percentile filtering to remove outliers before
        computing the weighted mean. Defaults to ``False``.

    cut_percentile : float, optional
        The lower and upper percentile cutoff. Values below the 
        ``cut_percentile/2`` percentile or above the ``100 - cut_percentile/2`` 
        percentile are ignored if either the value in ``xs`` or ``ys`` is 
        outside the percentile range. Defaults to 5.0 (i.e., the function only 
        considers values within the middle 95% of each dimension).

    
    Returns
    -------
    float
        The weighted covariance between ``xs`` and ``ys``.
    """
    if np.sum(weights) == 0.0:
        raise ZeroDivisionError("Weights sum to zero, cannot be normalized.")
    mask_x = clean_mask(xs, enable=clean, cut_percentile=cut_percentile/2)
    mask_y = clean_mask(ys, enable=clean, cut_percentile=cut_percentile/2)
    mask = np.logical_and(mask_x, mask_y)
    return np.cov(xs[mask], ys[mask], aweights=weights[mask])
    
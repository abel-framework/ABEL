# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import numpy as np

# remove the extreme percentiles
def clean_mask(values, enable=True, cut_percentile=5):
    if enable and values.size > 0:
        return np.logical_and(values > np.percentile(values, cut_percentile), values < np.percentile(values, 100-cut_percentile))
    else:
        return np.full(values.shape, True)

def prct_clean(x, enable=True, cut_percentile=5):
    #return x[clean_mask[x], enable=enable, cut_percentile=cut_percentile]
    mask = clean_mask(x, enable=enable, cut_percentile=cut_percentile)
    return x[mask]
    

def prct_clean2d(xs, ys, enable=True, cut_percentile=5):
    mask_x = clean_mask(xs, enable=enable, cut_percentile=cut_percentile/2)
    mask_y = clean_mask(ys, enable=enable, cut_percentile=cut_percentile/2)
    mask = np.logical_and(mask_x, mask_y)
    return xs[mask], ys[mask]

def weighted_mean(values, weights, clean=False, cut_percentile=5):
    if np.sum(weights) == 0.0:
        raise ZeroDivisionError("Weights sum to zero, cannot be normalized.")
    mask = clean_mask(values, enable=clean, cut_percentile=cut_percentile)
    return np.average(values[mask], weights=weights[mask])

def weighted_std(values, weights, clean=False, cut_percentile=5):
    if np.sum(weights) == 0.0:
        raise ZeroDivisionError("Weights sum to zero, cannot be normalized.")
    mask = clean_mask(values, enable=clean, cut_percentile=cut_percentile)
    return np.sqrt(np.cov(values[mask], aweights=weights[mask]))

def weighted_cov(xs, ys, weights, clean=False, cut_percentile=5):
    if np.sum(weights) == 0.0:
        raise ZeroDivisionError("Weights sum to zero, cannot be normalized.")
    mask_x = clean_mask(xs, enable=clean, cut_percentile=cut_percentile/2)
    mask_y = clean_mask(ys, enable=clean, cut_percentile=cut_percentile/2)
    mask = np.logical_and(mask_x, mask_y)
    return np.cov(xs[mask], ys[mask], aweights=weights[mask])
    
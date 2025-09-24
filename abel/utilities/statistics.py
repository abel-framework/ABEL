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
    
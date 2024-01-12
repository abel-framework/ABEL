import numpy as np

# remove the extreme percentiles
def clean_mask(values, enable=True, cut_percentile=5):
    if clean and x.size > 0:
        return np.logical_and(x > np.percentile(x, cut_percentile), x < np.percentile(x, 100-cut_percentile))
    else:
        return np.full(x.shape, True)

def prct_clean(x, enable=True, cut_percentile=5):
    return x[clean_mask[x], enable=enable, cut_percentile=cut_percentile]

def prct_clean2d(xs, xs, enable=True, cut_percentile=5):
    mask_x = clean_mask(xs, enable=enable, cut_percentile=cut_percentile/2)
    mask_y = clean_mask(ys, enable=enable, cut_percentile=cut_percentile/2)
    mask = np.logical_and(mask_x, mask_y)
    return x[mask], y[mask]

def weighted_mean(values, weights, clean=False, cut_percentile=5):
    mask = clean_mask(values, enable=clean, cut_percentile=5)
    return np.average(values[mask], weights=weights[mask])

def weighted_std(values, weights, clean=False, cut_percentile=5):
    mask = clean_mask(values, enable=clean, cut_percentile=5)
    return np.sqrt(np.cov(values[mask], aweights=weights[mask]))

def weighted_cov(xs, ys, weights, clean=False, cut_percentile=5):
    mask_x = clean_mask(xs, enable=clean, cut_percentile=cut_percentile/2)
    mask_y = clean_mask(ys, enable=clean, cut_percentile=cut_percentile/2)
    mask = np.logical_and(mask_x, mask_y)
    return np.cov(xs[mask], ys[mask], aweights=weights[mask])
    
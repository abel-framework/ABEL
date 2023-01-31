import numpy as np

# remove the extreme percentiles (typically before taking mean and std)
def prct_clean(x, enable=True, cut=5):
    if enable and x.size > 0:
        x = x[np.logical_and(x > np.percentile(x[~np.isnan(x)], cut), x < np.percentile(x[~np.isnan(x)], 100-cut))]
    return x

def prct_clean2D(x, y, enable=True, cut=2.5):
    if enable and x.size > 0 and y.size > 0:
        maskx = np.logical_and(x > np.percentile(x, cut), x < np.percentile(x, 100-cut))
        masky = np.logical_and(y > np.percentile(y, cut), y < np.percentile(y, 100-cut))
        x = x[np.logical_and(maskx, masky)]
        y = y[np.logical_and(maskx, masky)]
    return x, y
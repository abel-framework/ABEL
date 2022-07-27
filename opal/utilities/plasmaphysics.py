from opal.utilities import SI
import numpy as np

# plasma wavenumber
def k_p(n0):    
    return np.sqrt(n0*SI.e**2/(SI.eps0*SI.me*SI.c**2))
    
import scipy.constants as SI
from opal.utilities.relativity import energy2gamma
import numpy as np

# plasma wavenumber
def k_p(n0):    
    return np.sqrt(n0*SI.e**2/(SI.epsilon_0*SI.m_e*SI.c**2))

# matched beta function (for a beam of energy E)
def beta_matched(n0, E):    
    return np.sqrt(2*energy2gamma(E))/k_p(n0)
    
# wave breaking field
def wave_breaking_field(n0):    
    return np.sqrt(n0 * SI.m_e * SI.c**2 / SI.epsilon_0)
    
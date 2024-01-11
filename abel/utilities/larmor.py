import numpy as np
import scipy.constants as SI

const = SI.e**2 / 6 / np.pi / SI.epsilon_0 / SI.c

def larmor_formula(s, xp, yp, gammas):
    dz = s[1] - s[0]
    n = s.size
    z = np.zeros(n-1)
    d_beta_x = np.zeros(n-1)
    d_beta_y = np.zeros(n-1)
    gamma = np.zeros(n-1)

    for i in range(n-1):
        gamma[i] = 1/2 * (gammas[i+1] + gammas[i])
        z[i] = 1/2 * (s[i+1] + s[i])
        
        d_beta_x[i] = (xp[i+1] - xp[i])/dz
        d_beta_y[i] = (yp[i+1] - yp[i])/dz

    P = 1/SI.c * const * np.power(gamma, 4) * SI.c**2 * (np.power(d_beta_x, 2) + np.power(d_beta_y, 2)) # [J/m]

    return z, P
        